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

def apply_matrix_C( matrix_to_apply: Optional[torch.Tensor],list_grad_tensor: list,list_noise_tensor: list) -> torch.Tensor:

       
    reference_tensor = list_grad_tensor[-1]
    original_shape = reference_tensor.shape
    device = reference_tensor.device
    
    D = int(reference_tensor.numel())  # Counts the total number of elements in p.summed_grad 
    num_iterations = len(list_grad_tensor) # number of iterations so far 

    matrix_to_apply = matrix_to_apply[0:num_iterations,0:num_iterations] # # we need the right row at step t and Cast to double
    # print(matrix_to_apply)
    matrix_to_apply = matrix_to_apply.to(device) # make sure the matrix is in the right device as the grad_tensor

    flattened_gradients = []
    flattened_noise = []
    for grad_tensor in list_grad_tensor:
        if grad_tensor.numel() != D:
            raise ValueError(
                f"All tensors in list_grad_tensor must have the same number of elements ({D}). "
                f"Found a tensor with {grad_tensor.numel()} elements."
            )
        flattened_gradients.append(grad_tensor.detach().clone().view(D).to(device)) # Cast to double
    for grad_tensor in list_noise_tensor:
        if grad_tensor.numel() != D:
            raise ValueError(
                f"All tensors in list_grad_tensor must have the same number of elements ({D}). "
                f"Found a tensor with {grad_tensor.numel()} elements."
            )
        flattened_noise.append(grad_tensor.detach().clone().view(D).to(device)) # Cast to double

    stacked_gradients = torch.stack(flattened_gradients, dim=0)    # The shape of stacked_gradients will be (num_iterations, D).
    stacked_noises = torch.stack(flattened_noise, dim=0)  
    # Apply matrix
    with torch.no_grad():
        transformed = matrix_to_apply @ stacked_gradients  # (txt) @ (t,d) = (t,d)


    # Reshape back to original shape
    return (transformed + stacked_noises).float() # (txd) + (txd) = (t,d)

def apply_matrix_B( matrix_to_apply: Optional[torch.Tensor],grad_tensor) -> torch.Tensor:



    device = grad_tensor.device
    matrix_to_apply = matrix_to_apply.to(device)
    with torch.no_grad():
        transformed_flat = matrix_to_apply @ grad_tensor #(1xt) x(txd) =(1xd)

    # Reshape back to original shape
    return transformed_flat.float()
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
                A_matrix,
                B_matrix,
                C_matrix,
                 **kwargs):
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
        self.A_matrix = A_matrix
        self.B_matrix = B_matrix
        self.C_matrix = C_matrix
        self.historical_parameter_noises = []
    def add_noise(self):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """

        for p in self.params:
            _check_processed_flag(p.summed_grad)
            
            sens_C = compute_sensitivity(self.C_matrix)
            noise = _generate_noise(
                std=self.noise_multiplier * self.max_grad_norm*sens_C,
                reference=p.summed_grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            # # --- Initialize p.noise_history if it doesn't exist ---
            if not hasattr(p, 'noise_history'):
                p.noise_history = []

            # --- Initialize p.summed_grad_history if it doesn't exist ---
            if not hasattr(p, 'summed_grad_history'):
                p.summed_grad_history = []
            if not hasattr(p, 'Q_last'): #Q= AG_dash + sens(C)BZ
                p.Q_last = 0

            

            # --- Append a detached clone of the current noise to this parameter's history ---
            p.noise_history.append(noise.detach().clone())
            # --- Append a detached clone of the current clipped gradient to this parameter's history ---
            p.summed_grad_history.append(p.summed_grad.detach().clone())

        
            Matrix_applied_summed_grad_noise_added = apply_matrix_C(self.C_matrix,p.summed_grad_history,p.noise_history) #CG_dash +Z


            num_iterations = len(p.summed_grad_history) # number of iterations so far 

            matrix_to_apply = self.B_matrix[num_iterations-1] # # we need the right row at step t and Cast to double
            matrix_to_apply = matrix_to_apply[0:num_iterations] # then we take the first columns that exist in the lower triangle, cast to double
             # make sure the matrix is in the right device as the grad_tensor
            Q = apply_matrix_B(matrix_to_apply,Matrix_applied_summed_grad_noise_added).view(p.summed_grad_history[-1].shape).view_as(p)
            # #this should the p.grad that will be used in next steps hopefully
            p.grad = Q - p.Q_last


            # #save the last Q for next step computation
            p.Q_last = Q

            _mark_as_processed(p.summed_grad)


    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """
        Perform actions specific to ``DPOptimizer`` before calling
        underlying  ``optimizer.step()``

        Args:
            closure: A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        # The corner case when the optimizer has no trainable parameters.
        # Essentially the DPOptimizer act as a normal optimizer
        if self.grad_samples is None or len(self.grad_samples) == 0:
            return True
        self.clip_and_accumulate()
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False
        self.add_noise()
        self.scale_grad()
        if self.step_hook:
            self.step_hook(self)
        self._is_last_step_skipped = False
        return True