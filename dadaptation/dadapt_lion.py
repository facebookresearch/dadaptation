# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Optional, Callable

import torch
from torch.optim.optimizer import Optimizer
import torch.distributed as dist
import logging
import pdb

class DAdaptLion(Optimizer):
    r"""
    Implements Lion with D-Adaptation automatic step-sizes. 
    Has not been as heavily tested as DAdaptAdam and should be considered experimental.
    
    
    Leave LR set to 1 unless you encounter instability.
    Arguments:
        params (iterable): 
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): 
            Learning rate adjustment parameter. Increases or decreases the D-adapted learning rate.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        weight_decay (float): 
            Weight decay, i.e. a L2 penalty (default: 0).
        log_every (int): 
            Log using print every k steps, default 0 (no logging).
        d0 (float):
            Initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
        fsdp_in_use (bool):
            If you're using sharded parameters, this should be set to True. The optimizer
            will attempt to auto-detect this, but if you're using an implementation other
            than PyTorch's builtin version, the auto-detection won't work.
    """
    def __init__(
        self,
        params,
        lr: float = 1.0,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0, log_every=0,
        d0=1e-6, fsdp_in_use=False):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            d=d0, k=0,
            log_every=log_every,
            numerator_weighted=0.0,
            fsdp_in_use=fsdp_in_use)

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        numerator_weighted = group['numerator_weighted']
        d = group['d']
        lr = max(group['lr'] for group in self.param_groups)

        dlr = d*lr
        
        log_every = group['log_every']
        fsdp_in_use = group['fsdp_in_use']

        beta1, beta2 = group['betas']
        sqrt_beta2 = beta2**0.5

        numerator_acum = 0.0
        sk_l1 = 0.0

        for group in self.param_groups:
            k = group['k']
            group_lr = group['lr'] 
            wd = group['weight_decay']

            if group_lr not in [lr, 0.0]:
                raise RuntimeError(f"Setting different lr values in different parameter groups is only supported for values of 0")

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p).detach()
                    state['s'] = torch.zeros_like(p).detach()

                exp_avg = state['exp_avg']
                s = state['s']

                #AdamW style weight decay
                p.data.mul_(1-dlr*wd)

                update = exp_avg.clone().mul_(beta1).add_(grad, alpha=(1-beta1)).sign_()

                p.data.add_(update, alpha=-dlr)

                exp_avg.mul_(beta2).add_(grad, alpha=(1-beta2)*dlr)

                numerator_acum += dlr * torch.dot(update.flatten(), s.flatten()).item()
                
                s.mul_(sqrt_beta2).add_(update, alpha=(1-sqrt_beta2)*dlr)
                
                sk_l1 += s.abs().sum().item()

        numerator_weighted = sqrt_beta2*numerator_weighted + (1-sqrt_beta2)*numerator_acum
        d_hat = d
        
        # if we have not done any progres, return
        # if we have any gradients available, will have sk_l1 > 0 (unless \|g\|=0)
        if sk_l1 == 0:
            return loss
        
        if lr > 0.0:
            if fsdp_in_use:
                dist_tensor = torch.zeros(2).cuda()
                dist_tensor[0] = numerator_weighted
                dist_tensor[1] = sk_l1
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_numerator_weighted = dist_tensor[0]
                global_sk_l1 = dist_tensor[1]
            else:
                global_numerator_weighted = numerator_weighted
                global_sk_l1 = sk_l1


            d_hat = global_numerator_weighted/((1-sqrt_beta2)*global_sk_l1)
            d = max(d, d_hat)

        if log_every > 0 and k % log_every == 0:
            logging.info(f"lr: {lr} dlr: {dlr} d_hat: {d_hat}, d: {d}. sk_l1={global_sk_l1:1.1e} numerator_weighted={global_numerator_weighted:1.1e}")
        
        for group in self.param_groups:
            group['numerator_weighted'] = numerator_weighted
            group['d'] = d
            group['k'] = group['k'] + 1

        return loss