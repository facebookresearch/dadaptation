# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.optim
import pdb
import math
import logging
import torch.distributed as dist

class DAdaptSGD(torch.optim.Optimizer):
    r"""
    Implements SGD with D-Adaptation automatic step-sizes. Leave LR set to 1 unless you encounter instability.

    Arguments:
        params (iterable): 
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): 
            Learning rate adjustment parameter. Increases or decreases the D-adapted learning rate.
        momentum (float): 
            Momentum value in  the range [0,1) (default: 0).
        weight_decay (float): 
            Weight decay, i.e. a L2 penalty (default: 0).
        log_every (int): 
            Log using print every k steps, default 0 (no logging).
        d0 (float):
            Initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
        growth_rate (float):
            prevent the D estimate from growing faster than this multiplicative rate. 
            Default is inf, for unrestricted. More conservative values like 1.02 may
            help if training is unstable.
        fsdp_in_use (bool):
            If you're using sharded parameters, this should be set to True. The optimizer
            will attempt to auto-detect this, but if you're using an implementation other
            than PyTorch's builtin version, the auto-detection won't work.
    """
    def __init__(self, params, 
        lr=1.0, 
        momentum=0.0, 
        weight_decay=0, 
        log_every=0,
        d0=1e-6, growth_rate=float('inf'),
        fsdp_in_use=False):

        if not 0.0 < d0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr,
            momentum=momentum, 
            weight_decay=weight_decay, k=0,
            log_every=log_every,
            numerator_weighted=0.0, 
            d=d0, 
            growth_rate=growth_rate,
            fsdp_in_use=fsdp_in_use)
        self.loggables = {}

        try:
            self.rank = torch.distributed.get_rank()
        except:
            self.rank = 0

        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        group = self.param_groups[0]
        lr = max(group['lr'] for group in self.param_groups)

        decay = group['weight_decay']
        momentum = group['momentum']
        log_every = group['log_every']
        ck = 1 - momentum
        k = group['k']

        numerator_weighted = group['numerator_weighted']
        growth_rate = group['growth_rate']
        d = group['d']
        fsdp_in_use = group['fsdp_in_use']
        
        group = self.param_groups[0]
        
        sk_sq = 0.0

        if k == 0: 
            g_sq = 0.0
            for group in self.param_groups:
                group_lr = group['lr']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    if hasattr(p, "_fsdp_flattened"):
                        fsdp_in_use = True
                    grad = p.grad.data
                    
                    # Apply weight decay
                    if decay != 0:
                        grad.add(p.data, alpha=decay)

                    state = self.state[p]

                    if group_lr > 0.0:
                        g_sq += (grad * grad).sum().item()

            if fsdp_in_use:
                dist_tensor = torch.zeros(1).cuda()
                dist_tensor[0] = g_sq
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_gsq = dist_tensor[0]
            else:
                global_gsq = g_sq
            group['g0_norm'] = g0_norm = math.sqrt(global_gsq)

        g0_norm = group['g0_norm']

        dlr = d*lr/g0_norm

        for group in self.param_groups:
            group_lr = group['lr']
            if group_lr not in [lr, 0.0]:
                raise RuntimeError(f"Setting different lr values in different parameter groups is only supported for values of 0")
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if 'z' not in state:
                    z = state['z'] = torch.clone(p.data).detach()
                    s = state['s'] = torch.zeros_like(p.data).detach()
                    x0 = state['x0'] = torch.clone(p.data).detach()

                # Apply weight decay
                if decay != 0:
                    grad.add_(p.data, alpha=decay)

                s = state['s']

                if group_lr > 0.0:
                    numerator_weighted += dlr * torch.dot(grad.flatten(), s.flatten()).item()
                    
                    s.data.add_(grad, alpha=dlr)
                    sk_sq += (s * s).sum().item()
            ######

        d_hat = d

        if lr > 0.0:
            if fsdp_in_use:
                dist_tensor = torch.zeros(2).cuda()
                dist_tensor[0] = sk_sq
                dist_tensor[1] = numerator_weighted
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_sk_sq = dist_tensor[0]
                global_numerator_weighted = dist_tensor[1]
            else:
                global_sk_sq = sk_sq
                global_numerator_weighted = numerator_weighted

            d_hat = 2*global_numerator_weighted/math.sqrt(global_sk_sq)
            d = max(d, min(d_hat, d*growth_rate))


        # if we have not done any updates
        # if we have any gradients available, will have sk_sq > 0 (unless \|g\|=0)
        if global_sk_sq == 0:
            return loss

        if log_every > 0 and k % log_every == 0:
            logging.info(f"(r={self.rank},k={k}) dlr: {dlr} d_hat: {d_hat}, d: {d}. sk_norm={math.sqrt(global_sk_sq)} numerator_weighted={global_numerator_weighted} g0_norm={g0_norm}")

        for group in self.param_groups:
            group['numerator_weighted'] = numerator_weighted
            group['d'] = d
            group['g0_norm'] = g0_norm
            ######################################
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                s = state['s']
                x0 = state['x0']
                z = state['z']

                # z step
                z.data.copy_(x0 - s)

                # x step
                p.data.mul_(1-ck).add_(z, alpha=ck)

            group['k'] = k + 1

        return loss