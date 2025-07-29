# _create_noisy_clipped_dice_gradient method from: https://github.com/564612540/DiceSGD/blob/master/DiceSGD/optimizers_utils.py
from typing import Optional, Callable
import torch
import math
from torch.optim import Optimizer
from opacus.optimizers import DPOptimizer
from itertools import chain

# from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable, List, Optional, Union

import torch
from opacus.optimizers.utils import params
from torch import nn
from torch.optim import Optimizer
from opacus.optimizers.optimizer import _mark_as_processed, _check_processed_flag_tensor,_check_processed_flag,_generate_noise

logger = logging.getLogger(__name__)
logger.disabled = True






class DPOptimizer_Dice(DPOptimizer):
    def __init__(
                self,
                optimizer: Optimizer,
                *,
                noise_multiplier: float,
                max_grad_norm: float,
                expected_batch_size: Optional[int],
                loss_reduction: str = "mean",
                generator=None,
                secure_mode: bool = False,
                normalize_clipping, 
                error_max_grad_norm=1.0):
        super().__init__(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            expected_batch_size=expected_batch_size,
            max_grad_norm=max_grad_norm,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            normalize_clipping=normalize_clipping
        )
        self.error_max_grad_norm = error_max_grad_norm

    def add_noise(self):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """

        max_grad_norm = 1 if self.normalize_clipping else self.max_grad_norm

        # __________________________New Code Start _______
        signals, noises, error_norms, grad_norms = [], [], [], []

        for p in self.params:
                if p.requires_grad:
                    if hasattr(p,'error'):
                        first_minibatch = False
                        error_norms.append(p.error.reshape(-1).norm(2))
                    else:
                        # param.error = None
                        first_minibatch = True
                        error_norms.append(torch.tensor(0.))
        error_norm = torch.stack(error_norms).norm(2) + 1e-6

        # __________________________New Code End _______
        for p in self.params:
            _check_processed_flag(p.summed_grad)    
            # __________________________New Code Start _______
            grad_diff = (p.grad-p.summed_grad)
            grad_dummy = p.grad

            p.grad = p.summed_grad  # Ultra important to override `.grad`.
            # del p.summed_grad  ----> This will be NONE using the zero_grad method in DP optimizer

            if first_minibatch:
                p.error=grad_diff
            else:
                #I used error_max_grad_norm as C2 instead of C2/C1 as they did in jax
                # They added a scaling by clipping factor that is not aligned with the paper's algoirthm whole updating p.error

                # Their implementation
                # p.grad += p.error*torch.clamp_max(max_grad_norm/error_norm*self.error_max_grad_norm,1) # vt ---> Their implementation
                # p.error=p.error*torch.clamp_max(self.error_max_grad_norm\
                #                                 (1-torch.clamp_max(self.error_max_grad_norm/error_norm,\
                #                                                            1)))\
                #                                 +grad_diff   
        

                #My implementation
                p.grad += p.error*torch.clamp_max(self.error_max_grad_norm/error_norm,1)  #same as their implementation
                p.error=p.error+ grad_dummy -p.grad  #Same as the paper (not exactly the same as theirs)


                # in the paper
                # #e = e + g - v                                    # my implementation
                #    = e +g -g_clipped - e.clip(e,1)
                #    = e(1-clip_factor) + g-g_clipped
                #    = e*(1-clip_factor)  + grad_diff

                #    = e*clip_factor*(1-clip_factor) + grad_diff    # their implementation ( should not be correct)
                            
            del grad_diff

            # __________________________New Code End _______
            noise = _generate_noise(
                # Note they use error_max_grad_norm as C2/C1. my error_max_grad_norm = C2. So I will mimic 
                std=self.noise_multiplier * max_grad_norm * math.sqrt(1+2* (self.error_max_grad_norm/max_grad_norm)),
                reference=p.grad, # ---- changed from   summed_grad to grad
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            # p.grad = (p.summed_grad + noise).view_as(p) ---> removed from actual code
            # __________________________New Code Start _______
            p.grad += noise #vt + wt --->  from the paper

            # __________________________New Code End _______

            _mark_as_processed(p.summed_grad)


