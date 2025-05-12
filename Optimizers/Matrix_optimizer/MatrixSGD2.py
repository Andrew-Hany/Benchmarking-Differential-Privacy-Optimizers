from typing import Optional, Callable
import torch
import math
from torch.optim import Optimizer
from opacus.optimizers import DPOptimizer
from itertools import chain
import logging
from collections import defaultdict
from typing import Callable, List, Optional, Union

import torch
from opacus.optimizers.utils import params
from torch import nn
from torch.optim import Optimizer
import gc
import numpy as np

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


    # Apply matrix
    with torch.no_grad():
        transformed_flat = matrix_to_apply @ stacked_gradients


    del matrix_to_apply
    del stacked_gradients

    # Reshape back to original shape
    return transformed_flat.float().view(original_shape)

def gpu_matrix_multiply(T:int,D:int,a_path: str, b_path: str, tile_size: int = 10240 ) -> torch.Tensor:
    """
    Performs matrix multiplication on the GPU using PyTorch, handling large matrices
    that may not fit in GPU memory by tiling.
    Loads matrices from disk, not from RAM.

    Args:
        a_path: Path to the first matrix (A) saved as a .pt file.
        b_path: Path to the second matrix (B) saved as a .pt file.
        tile_size: Size of the square tiles to use for processing on the GPU.

    Returns:
        Result of the matrix multiplication as a PyTorch tensor on the CPU.
    """

    b_np = np.load(a_path, mmap_mode='r')
    z_np = np.load(b_path, mmap_mode='r')

    result_cpu = torch.zeros((T, D), dtype=torch.float32, device="cpu")
    print('passed')

    num_tiles_m = math.ceil(T / tile_size)
    num_tiles_k = math.ceil(T / tile_size)
    num_tiles_n = math.ceil(D / tile_size)

    print(f"Matrices shape: B({T}x{T}), Z({T}x{D})")
    print(f"Using tile size: {tile_size}")
    print(f"Number of tiles: M={num_tiles_m}, K={num_tiles_k}, N={num_tiles_n}")

    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and PyTorch with CUDA support installed.")

    # Move the entire matrices to the CPU (already done above)
    # Now process each tile on the GPU

    for i in range(num_tiles_m):
        for j in range(num_tiles_n):
            c_row_start = i * tile_size
            c_row_end = min((i + 1) * tile_size, T)
            c_col_start = j * tile_size
            c_col_end = min((j + 1) * tile_size, D)

            for l in range(num_tiles_k):
                a_row_start = i * tile_size
                a_row_end = min((i + 1) * tile_size, T)
                a_col_start = l * tile_size
                a_col_end = min((l + 1) * tile_size, T)

                b_row_start = l * tile_size
                b_row_end = min((l + 1) * tile_size, T)
                b_col_start = j * tile_size
                b_col_end = min((j + 1) * tile_size, D)

                # Extract tile from CPU
                # Load the tile B_il from b_np
                a_tile_cpu = b_np[a_row_start:a_row_end, a_col_start:a_col_end].copy()
                # Load the tile Z_lj from z_np
                b_tile_cpu = z_np[b_row_start:b_row_end, b_col_start:b_col_end].copy()

                print((i * num_tiles_n * num_tiles_k) + (j * num_tiles_k) + l,'/',num_tiles_m*num_tiles_k*num_tiles_n)
                # Move tile to GPU
                a_tile_gpu = torch.from_numpy(a_tile_cpu).to("cuda")
                b_tile_gpu = torch.from_numpy(b_tile_cpu).to("cuda")

                # Perform matrix multiplication on GPU
                c_tile_gpu = a_tile_gpu @ b_tile_gpu

                # Move result back to CPU
                c_tile_cpu = c_tile_gpu.cpu()

                # Accumulate result
                result_cpu[c_row_start:c_row_end, c_col_start:c_col_end] += c_tile_cpu

                # Free up memory
                del  a_tile_cpu,b_tile_cpu,a_tile_gpu, b_tile_gpu, c_tile_gpu, c_tile_cpu
                torch.cuda.empty_cache()
                gc.collect()

    return result_cpu
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
                # A_matrix,
                B_matrix,
                sens_C,
                num_steps,
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
        # self.A_matrix = A_matrix
        self.B_matrix = B_matrix
        self.sens_C = sens_C
        self.num_steps =  num_steps
        self.noise_history = None  # Will hold the precomputed noise for all steps
        self.correlated_noise_history = None
        self.current_step = 0      # Track which step we're on


    def _generate_all_noise(self):
        """
        Precomputes correlated noise for all steps and stores it in self.correlated_noise_history.
        Deletes all intermediate tensors to save memory.
        """

        device = 'cpu'

        # Step 1: Generate raw noise for each step and parameter
        noise_per_step = []
        for _ in range(self.num_steps):
            step_noise = []
            for p in self.params:
                noise = _generate_noise(
                    std=self.noise_multiplier * self.max_grad_norm,
                    reference=p.summed_grad,
                    generator=self.generator,
                    secure_mode=self.secure_mode,
                )
                step_noise.append(noise.to('cpu'))
                del noise
            noise_per_step.append(step_noise)
        print('step 1 is done')
        # Step 2: Flatten all parameters' noise into a single vector per step
        T = self.num_steps
        D = sum(p.numel() for p in self.params)  # Total number of elements across all parameters
        Z = torch.zeros(T, D, device=device)

        for step_idx in range(T):
            flat_noise = torch.cat([n.view(-1) for n in noise_per_step[step_idx]])
            Z[step_idx] = flat_noise

        del noise_per_step
        print('step 2 is done')
        # Step 3: Multiply with B (T x T)
        # self.B_matrix = self.B_matrix.to(device)
        np.save("B_matrix.npy", self.B_matrix.numpy())
        np.save("Z.npy", Z.numpy())

        # Delete from RAM
        del self.B_matrix
        del Z
        gc.collect()
        torch.cuda.empty_cache()

        print('passed')
        correlated_Z = gpu_matrix_multiply(T,D,"B_matrix.npy", "Z.npy")


        print('step 3 is done')
        # Step 4: Reshape back to per-parameter structure
        self.correlated_noise_history = []

        for step_idx in range(T):
            flat_correlated = correlated_Z[step_idx]
            current_step_correlated = []

            idx = 0
            for p in self.params:
                numel = p.numel()
                current_step_correlated.append(flat_correlated[idx:idx + numel].view(p.shape))
                idx += numel

            self.correlated_noise_history.append(current_step_correlated)

        print('step 4 is done')
        # Step 5: Clean up memory
        del correlated_Z
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                


    def add_noise(self):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """
        
        # If no noise history exists, generate it now
        device = next(iter(self.params)).device
        if self.correlated_noise_history is None:
            print('Generating Noise start ...')
            print('Pre-generating all the noise -- It might take some time but next steps will not')
            self._generate_all_noise()
            print('generation is done')

        # Get the noise for the current step
        step__noise_correlated = self.correlated_noise_history[self.current_step] #BZ
        # Move the correlated noise to the same device as the model
        step__noise_correlated = [n.to(device) for n in step__noise_correlated]
        for i, p in enumerate(self.params):
            _check_processed_flag(p.summed_grad)
            
            # # --- Initialize p.noise_history if it doesn't exist ---
            if not hasattr(p, 'noise_last'):
                p.noise_last = torch.zeros_like(step__noise_correlated[i])

            p.grad = (p.summed_grad - p.noise_last + step__noise_correlated[i]).view_as(p)
            p.noise_last = step__noise_correlated[i]
            # p.grad = (p.summed_grad + noise).view_as(p)
            # p.grad = (p.summed_grad + step_correlated[i]).view_as(p)
            _mark_as_processed(p.summed_grad)
        del step__noise_correlated
        self.current_step += 1


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