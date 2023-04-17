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
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        momentum (float): 
            Momentum value in  the range [0,1) (default: 0.9).
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
        momentum=0, 
        weight_decay=0, 
        log_every=0,
        d0=1e-6, growth_rate=float('inf'),
        fsdp_in_use=False):

        if d0 <= 0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if lr <= 0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr,
            momentum=momentum, 
            weight_decay=weight_decay, 
            k=0,
            log_every=log_every,
            gsq_weighted=0.0, 
            g0_norm=0.0,
            d=d0, 
            growth_rate=growth_rate,
            fsdp_in_use=fsdp_in_use)
        self.loggables = {}

        try:
            self.rank = torch.distributed.get_rank()
        except:
            self.rank = 0

        #self.group = dist.new_group(range(dist.get_world_size()))

        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        lr = max(group['lr'] for group in self.param_groups)

        # Stored at the group level to ensure resuming from checkpoints works
        group = self.param_groups[0]
        decay = group['weight_decay']
        momentum = group['momentum']
        log_every = group['log_every']
        ck = 1 - momentum
        k = group['k']

        gsq_weighted = group['gsq_weighted']
        g0_norm = group['g0_norm']
        growth_rate = group['growth_rate']
        fsdp_in_use = group['fsdp_in_use']
        d = group['d']

        g_sq = 0.0
        sk_sq = 0.0

        for group in self.param_groups:
            group_lr = group['lr']
            if group_lr not in [lr, 0.0]:
                raise RuntimeError(f"Setting different lr values in different parameter groups is only supported for values of 0")
            for p in group['params']:
                if p.grad is None:
                    continue
                if hasattr(p, "_fsdp_flattened"):
                    fsdp_in_use = True
                grad = p.grad.data
                
                # Apply weight decay
                if decay != 0:
                    if grad.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients")

                    grad.add_(p.data, alpha=decay)

                state = self.state[p]

                if group_lr > 0.0:
                    g_sq += (grad * grad).sum().item()

        # if we have any gradients available, will have g_sq > 0 (unless \|g\|=0)
        if g_sq == 0:
            return loss

        if k == 0: 
            if fsdp_in_use:
                dist_tensor = torch.zeros(1).cuda()
                dist_tensor[0] = g_sq
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_gsq = dist_tensor[0]
            else:
                global_gsq = g_sq
            g0_norm = math.sqrt(global_gsq)

        dlr = d*lr/g0_norm

        for group in self.param_groups:
            group_lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if 'z' not in state:
                    z = state['z'] = torch.clone(p.data).detach()
                    s = state['s'] = torch.zeros_like(p.data).detach()
                    x0 = state['x0'] = torch.clone(p.data).detach()

                s = state['s']

                if group_lr > 0.0:
                    s.data.add_(grad, alpha=dlr)
                
                sk_sq += (s * s).sum().item()
            ######

        gsq_weighted = gsq_weighted + dlr*dlr*g_sq
        d_hat = d

        if lr > 0.0:
            if fsdp_in_use:
                dist_tensor = torch.zeros(2).cuda()
                dist_tensor[0] = sk_sq
                dist_tensor[1] = gsq_weighted
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_sk_sq = dist_tensor[0]
                global_gsq_weighted = dist_tensor[1]
            else:
                global_sk_sq = sk_sq
                global_gsq_weighted = gsq_weighted

            d_hat = (global_sk_sq - global_gsq_weighted)/(math.sqrt(global_sk_sq))
            d = group['d'] = max(d, min(d_hat, d*growth_rate))

        if log_every > 0 and k % log_every == 0:
            logging.info(
                f"(k={k}) dlr: {dlr:1.1e} d_hat: {d_hat:1.1e}, d: {d:1.8}. "
                f"sk_sq={global_sk_sq:1.1e} gsq_weighted={global_gsq_weighted:1.1e} "
                f"g0_norm={g0_norm:1.1e}{' (FSDP)' if fsdp_in_use else ''}")

        for group in self.param_groups:
            group['gsq_weighted'] = gsq_weighted
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

