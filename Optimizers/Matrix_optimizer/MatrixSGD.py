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


logger = logging.getLogger(__name__)
logger.disabled = True


def _mark_as_processed(obj: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Marks parameters that have already been used in the optimizer step.

    DP-SGD puts certain restrictions on how gradients can be accumulated. In particular,
    no gradient can be used twice - client must call .zero_grad() between
    optimizer steps, otherwise privacy guarantees are compromised.
    This method marks tensors that have already been used in optimizer steps to then
    check if zero_grad has been duly called.

    Notes:
          This is used to only mark ``p.grad_sample`` and ``p.summed_grad``

    Args:
        obj: tensor or a list of tensors to be marked
    """

    if isinstance(obj, torch.Tensor):
        obj._processed = True
    elif isinstance(obj, list):
        for x in obj:
            x._processed = True
def _check_processed_flag_tensor(x: torch.Tensor):
    """
    Checks if this gradient tensor has been previously used in optimization step.

    See Also:
        :meth:`~opacus.optimizers.optimizer._mark_as_processed`

    Args:
        x: gradient tensor

    Raises:
        ValueError
            If tensor has attribute ``._processed`` previously set by
            ``_mark_as_processed`` method
    """

    if hasattr(x, "_processed"):
        raise ValueError(
            "Gradients haven't been cleared since the last optimizer step. "
            "In order to obtain privacy guarantees you must call optimizer.zero_grad()"
            "on each step"
        )
def _check_processed_flag(obj: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Checks if this gradient tensor (or a list of tensors) has been previously
    used in optimization step.

    See Also:
        :meth:`~opacus.optimizers.optimizer._mark_as_processed`

    Args:
        x: gradient tensor or a list of tensors

    Raises:
        ValueError
            If tensor (or at least one tensor from the list) has attribute
            ``._processed`` previously set by ``_mark_as_processed`` method
    """

    if isinstance(obj, torch.Tensor):
        _check_processed_flag_tensor(obj)
    elif isinstance(obj, list):
        for x in obj:
            _check_processed_flag_tensor(x)


def _generate_noise(
    std: float,
    reference: torch.Tensor,
    generator=None,
    secure_mode: bool = False,
) -> torch.Tensor:
    """
    Generates noise according to a Gaussian distribution with mean 0

    Args:
        std: Standard deviation of the noise
        reference: The reference Tensor to get the appropriate shape and device
            for generating the noise
        generator: The PyTorch noise generator
        secure_mode: boolean showing if "secure" noise need to be generated
            (see the notes)

    Notes:
        If `secure_mode` is enabled, the generated noise is also secure
        against the floating point representation attacks, such as the ones
        in https://arxiv.org/abs/2107.10138 and https://arxiv.org/abs/2112.05307.
        The attack for Opacus first appeared in https://arxiv.org/abs/2112.05307.
        The implemented fix is based on https://arxiv.org/abs/2107.10138 and is
        achieved through calling the Gaussian noise function 2*n times, when n=2
        (see section 5.1 in https://arxiv.org/abs/2107.10138).

        Reason for choosing n=2: n can be any number > 1. The bigger, the more
        computation needs to be done (`2n` Gaussian samples will be generated).
        The reason we chose `n=2` is that, `n=1` could be easy to break and `n>2`
        is not really necessary. The complexity of the attack is `2^p(2n-1)`.
        In PyTorch, `p=53` and so complexity is `2^53(2n-1)`. With `n=1`, we get
        `2^53` (easy to break) but with `n=2`, we get `2^159`, which is hard
        enough for an attacker to break.
    """
    zeros = torch.zeros(reference.shape, device=reference.device)
    if std == 0:
        return zeros
    # TODO: handle device transfers: generator and reference tensor
    # could be on different devices
    if secure_mode:
        torch.normal(
            mean=0,
            std=std,
            size=(1, 1),
            device=reference.device,
            generator=generator,
        )  # generate, but throw away first generated Gaussian sample
        sum = zeros
        for _ in range(4):
            sum += torch.normal(
                mean=0,
                std=std,
                size=reference.shape,
                device=reference.device,
                generator=generator,
            )
        return sum / 2
    else:
        return torch.normal(
            mean=0,
            std=std,
            size=reference.shape,
            device=reference.device,
            generator=generator,
        )

def compute_sensitivity(C):
    if C is None:
        return 1.0
    return torch.linalg.norm(C, ord=2).item()

def apply_matrix(grad_tensor, B):
    """
    Applies matrix B to grad_tensor.
    
    Args:
        grad_tensor: PyTorch tensor (e.g., p.grad)
        B: Square matrix of size (D, D), where D = number of elements in grad_tensor
    
    Returns:
        transformed_tensor: Same shape as grad_tensor, after applying B
    """
    original_shape = grad_tensor.shape
    D = int(torch.numel(grad_tensor))  # Counts the total number of elements in p.summed_grad
    g_flat = grad_tensor.view(D)       # Flatten gradient

    # Apply matrix B: B @ g_flat
    # or 
    # Apply matrix C: C @ g_flat
    with torch.no_grad():
        transformed_flat = B @ g_flat

    # Reshape back to original shape
    return transformed_flat.view(original_shape)
class DPOptimizer_Matrix(DPOptimizer):
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
                B,
                C):
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
        self.B = B
        self.C = C

    def add_noise(self):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """

        for p in self.params:
            _check_processed_flag(p.summed_grad)
            
            sens_C = compute_sensitivity(self.C)

            noise = _generate_noise(
                std=self.noise_multiplier * self.max_grad_norm * sens_C,
                reference=p.summed_grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            p.grad = (p.summed_grad + noise).view_as(p)

            _mark_as_processed(p.summed_grad)
    
    
    #B. (C⋅G dash+sens(C)⋅Z)
    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        self.clip_and_accumulate()
        
        if self._check_skip_next_step():
            return False

        # Step 1: Apply matrix C to the clipped gradient
        #   From clip_and_accumulate function: 
        #   Stores clipped and aggregated gradients into `p.summed_grad```
        for p in self.params:
            if not getattr(p, "requires_grad", False):
                continue

            if p.summed_grad is not None:
                p.summed_grad = apply_matrix(p.summed_grad, C)

        # Step 2: Add noise to the already transformed gradient
        self.add_noise()  # Already adds sens(C) * Z
        # Step 3: Apply matrix B to final noisy gradient
        #   From add_noise function: 
        #   Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        for p in self.params:
            if not getattr(p, "requires_grad", False):
                continue
            if p.grad is not None:
                p.grad = apply_matrix(p.grad, B)


        
        self.scale_grad()

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True