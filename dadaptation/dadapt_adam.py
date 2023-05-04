# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
import torch.optim
import pdb
import logging
import os
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any

class DAdaptAdam(torch.optim.Optimizer):
    r"""
    Implements Adam with D-Adaptation automatic step-sizes. 
    Leave LR set to 1 unless you encounter instability.
    
    Arguments:
        params (iterable): 
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): 
            Learning rate adjustment parameter. Increases or decreases the D-adapted learning rate.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float): 
            Term added to the denominator outside of the root operation to improve numerical stability. (default: 0).
        weight_decay (float): 
            Weight decay, i.e. a L2 penalty (default: 0).
        log_every (int): 
            Log using print every k steps, default 0 (no logging).
        decouple (boolean): 
            Use AdamW style decoupled weight decay
        use_bias_correction (boolean):
            Turn on Adam's bias correction. Off by default.
        d0 (float):
            Initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
        growth_rate (float):
            prevent the D estimate from growing faster than this multiplicative rate. 
            Default is inf, for unrestricted. Values like 1.02 give a kind of learning
            rate warmup effect.
        fsdp_in_use (bool):
            If you're using sharded parameters, this should be set to True. The optimizer
            will attempt to auto-detect this, but if you're using an implementation other
            than PyTorch's builtin version, the auto-detection won't work.
    """
    def __init__(self, params, lr=1.0, 
                 betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, log_every=0,
                 decouple=False,
                 use_bias_correction=False,
                 d0=1e-6, growth_rate=float('inf'),
                 fsdp_in_use=False):
        if not 0.0 < d0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        if decouple:
            print(f"Using decoupled weight decay")

        
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        d = d0, 
                        k=0, 
                        numerator_weighted=0.0,
                        log_every=log_every,
                        growth_rate=growth_rate,
                        use_bias_correction=use_bias_correction,
                        decouple=decouple,
                        fsdp_in_use=fsdp_in_use)
        self.d0 = d0
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        sk_l1 = 0.0

        group = self.param_groups[0]
        use_bias_correction = group['use_bias_correction']
        numerator_weighted = group['numerator_weighted']
        beta1, beta2 = group['betas']
        k = group['k']

        d = group['d']
        lr = max(group['lr'] for group in self.param_groups)

        if use_bias_correction:
            bias_correction = ((1-beta2**(k+1))**0.5)/(1-beta1**(k+1))
        else:
            bias_correction = 1

        dlr = d*lr*bias_correction
        
        growth_rate = group['growth_rate']
        decouple = group['decouple']
        log_every = group['log_every']
        fsdp_in_use = group['fsdp_in_use']

        
        sqrt_beta2 = beta2**(0.5)

        numerator_acum = 0.0

        for group in self.param_groups:
            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']
            group_lr = group['lr']

            if group_lr not in [lr, 0.0]:
                raise RuntimeError(f"Setting different lr values in different parameter groups is only supported for values of 0")

            for p in group['params']:
                if p.grad is None:
                    continue
                if hasattr(p, "_fsdp_flattened"):
                    fsdp_in_use = True
                
                grad = p.grad.data
                
                # Apply weight decay (coupled variant)
                if decay != 0 and not decouple:
                    grad.add_(p.data, alpha=decay)

                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    state['s'] = torch.zeros_like(p.data).detach()
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data).detach()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data).detach()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                s = state['s']

                if group_lr > 0.0:
                    denom = exp_avg_sq.sqrt().add_(eps)
                    numerator_acum += dlr * torch.dot(grad.flatten(), s.div(denom).flatten()).item()

                    # Adam EMA updates
                    exp_avg.mul_(beta1).add_(grad, alpha=dlr*(1-beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                    s.mul_(sqrt_beta2).add_(grad, alpha=dlr*(1-sqrt_beta2))
                    sk_l1 += s.abs().sum().item()

            ######

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
            d = max(d, min(d_hat, d*growth_rate))

        if log_every > 0 and k % log_every == 0:
            logging.info(f"lr: {lr} dlr: {dlr} d_hat: {d_hat}, d: {d}. sk_l1={global_sk_l1:1.1e} numerator_weighted={global_numerator_weighted:1.1e}")

        for group in self.param_groups:
            group['numerator_weighted'] = numerator_weighted
            group['d'] = d

            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1

                denom = exp_avg_sq.sqrt().add_(eps)

                # Apply weight decay (decoupled variant)
                if decay != 0 and decouple:
                    p.data.add_(p.data, alpha=-decay * dlr)


                ### Take step
                p.data.addcdiv_(exp_avg, denom, value=-1)

            group['k'] = k + 1

        return loss
