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
import tqdm
import os

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

def gpu_matrix_multiply(T: int, D: int, a_path: str, b_path: str, out_path: str, tile_size: int = 16384) -> None:
    """
    Performs matrix multiplication on the GPU using PyTorch, handling large matrices
    by tiling. Loads input matrices (.npy) using memory mapping and writes the
    result directly to a memory-mapped file (raw format). Simplified version
    with minimal validation.

    Args:
        T: Rows of A, Columns of A, Rows of B.
        D: Columns of B.
        a_path: Path to the first matrix (A) saved as a .npy file.
        b_path: Path to the second matrix (B) saved as a .npy file.
        out_path: Path to save the resulting matrix (raw binary format).
        tile_size: Size of the square tiles to use for processing on the GPU.

    Returns:
        None. Writes the result to out_path.
    """

    # --- Load input matrices using memory mapping ---
    # Assumes files exist and are correct .npy format/shape
    a_np = np.load(a_path, mmap_mode='r')
    # b_np = np.load(b_path, mmap_mode='r')
    b_np = np.memmap(b_path, dtype=np.float32, mode='r', shape=(T, D))

    # --- Create memory-mapped array for the result ---
    # Assumes directory exists and write permissions are okay
    # 'w+' mode creates or truncates the file.
    result_mm = np.memmap(out_path, dtype=np.float32, mode='w+', shape=(T, D))
    # Initialize the output file to zeros.
    result_mm[:] = 0.0
    result_mm.flush()

    num_tiles_m = math.ceil(T / tile_size)
    num_tiles_k = math.ceil(T / tile_size)
    num_tiles_n = math.ceil(D / tile_size)

    print(f"Matrices shape: A({T}x{T}), B({T}x{D}) -> Result({T}x{D})")
    print(f"Using tile size: {tile_size}")
    print(f"Number of tiles: M={num_tiles_m}, K={num_tiles_k}, N={num_tiles_n}")
    print(f"Outputting raw result to: {out_path}")

    if not torch.cuda.is_available():
        # Removed cleanup logic here as requested
        raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and PyTorch with CUDA support installed.")



    # --- Tiled Multiplication ---
    for i in range(num_tiles_m):  # Tile rows of A/Result
        for j in range(num_tiles_n):  # Tile columns of B/Result
            c_row_start = i * tile_size
            c_row_end = min((i + 1) * tile_size, T)
            c_col_start = j * tile_size
            c_col_end = min((j + 1) * tile_size, D)

            for k in range(num_tiles_k):  # Tile cols of A and rows of B (shared dimension)
                a_row_start = i * tile_size
                a_row_end = min((i + 1) * tile_size, T)
                a_col_start = k * tile_size
                a_col_end = min((k + 1) * tile_size, T)

                b_row_start = k * tile_size
                b_row_end = min((k + 1) * tile_size, T)
                b_col_start = j * tile_size
                b_col_end = min((j + 1) * tile_size, D)


                print((i * num_tiles_n * num_tiles_k) + (j * num_tiles_k) + k,'/',num_tiles_m*num_tiles_k*num_tiles_n)
                # Extract tiles from input memory maps
                a_tile_cpu_np = a_np[a_row_start:a_row_end, a_col_start:a_col_end]
                b_tile_cpu_np = b_np[b_row_start:b_row_end, b_col_start:b_col_end].copy()

                # Skip if tiles ended up empty (can happen at edges)
                if a_tile_cpu_np.size == 0 or b_tile_cpu_np.size == 0:
                    continue

                # Use pin_memory for faster transfer
                a_tile_gpu = torch.from_numpy(a_tile_cpu_np.astype(np.float32)).pin_memory().to("cuda")
                b_tile_gpu = torch.from_numpy(b_tile_cpu_np.astype(np.float32)).pin_memory().to("cuda")

                print('transfer __done')

                # Perform multiplication on GPU
                c_tile_gpu = a_tile_gpu @ b_tile_gpu
                torch.cuda.synchronize()

                # Move result tile back to CPU
                c_tile_cpu_tensor = c_tile_gpu.cpu()
                c_tile_cpu_np = c_tile_cpu_tensor.numpy() # Convert to NumPy array

                # Accumulate result tile into the output memory-mapped file
                result_mm[c_row_start:c_row_end, c_col_start:c_col_end] += c_tile_cpu_np

                # Free up memory proactively
            #     del a_tile_cpu_np, b_tile_cpu_np, a_tile_gpu, b_tile_gpu, c_tile_gpu, c_tile_cpu_tensor, c_tile_cpu_np
            #     torch.cuda.empty_cache()
            # gc.collect() # Usually not needed

    # --- Finalization ---
    # Ensure all writes are flushed to disk
    result_mm.flush()
    print(f"Result flushed to: {out_path}")
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

    def get_correlated_step(self,step_idx, T, D, memmap_path="correlated_Z.raw"):
        # Load the full matrix as a memory-mapped array
        correlated_Z = np.memmap(memmap_path, dtype=np.float32, mode='r', shape=(T, D))
        
        # Get the row corresponding to this step
        flat_tensor = correlated_Z[step_idx].copy()  # Copy to avoid issues with view

        return flat_tensor

    # Then reshape it into parameter shapes
    def reshape_flat_to_params(self,flat_tensor, params):
        current_step = []
        idx = 0
        for p in params:
            numel = p.numel()
            tensor = flat_tensor[idx:idx + numel].copy()  # Copy to NumPy
            tensor = torch.from_numpy(tensor).view(p.shape)  # Convert to PyTorch
            current_step.append(tensor)
            idx += numel
        return current_step
        
    # def _generate_all_noise(self):
    #     """
    #     Precomputes correlated noise for all steps and stores it in self.correlated_noise_history.
    #     Deletes all intermediate tensors to save memory.
    #     """

    #     device = 'cpu'

    #     # Step 1: Generate raw noise for each step and parameter
    #     noise_per_step = []
    #     for _ in tqdm.tqdm(range(self.num_steps), desc='raw noise'):
    #         step_noise = []
    #         for p in self.params:
    #             noise = _generate_noise(
    #                 std=self.noise_multiplier * self.max_grad_norm,
    #                 reference=p.summed_grad,
    #                 generator=self.generator,
    #                 secure_mode=self.secure_mode,
    #             )
    #             step_noise.append(noise.to('cpu'))
    #             del noise
    #         noise_per_step.append(step_noise)
    #         # del step_noise
    #         # gc.collect()
    #     print('step 1 is done')
    #     # Step 2: Flatten all parameters' noise into a single vector per step
    #     T = self.num_steps
    #     D = sum(p.numel() for p in self.params)  # Total number of elements across all parameters
    #     Z = torch.zeros(T, D, device=device)

    #     for step_idx in range(T):
    #         flat_noise = torch.cat([n.view(-1) for n in noise_per_step[step_idx]])
    #         Z[step_idx] = flat_noise

    #     del noise_per_step
    #     print('step 2 is done')
    #     # Step 3: Multiply with B (T x T)
    #     np.save("B_matrix.npy", self.B_matrix.numpy())
    #     np.save("Z.npy", Z.numpy())

    #     # Delete from RAM
    #     del self.B_matrix
    #     del Z
    #     gc.collect()
    #     torch.cuda.empty_cache()

    #     gpu_matrix_multiply(T,D,"B_matrix.npy", "Z.npy",'correlated_Z.raw')
  
    #     print('step 3 is done')



    def _generate_all_noise(self):
        """
        Precomputes correlated noise for all steps and stores it in a memory-mapped file.
        Uses memory-mapped arrays to avoid loading everything into RAM.
        """

        # Step 1: Define T and D
        T = self.num_steps
        D = sum(p.numel() for p in self.params)
        print(T,D)
        # Step 2: Define file paths
        # noise_file = "noise_output.raw"
        B_file = "B_matrix.npy"
        Z_file = "Z.npy"
        result_file = "correlated_Z.raw"

        # Step 3: Create memory-mapped array for noise (T x D)


        Z_mm = np.memmap(Z_file, dtype=np.float32, mode='w+', shape=(T, D))
        Z_mm[:] = 0.0
        Z_mm.flush()

        # Step 4: Generate and save noise incrementally
        
        for step_idx in tqdm.tqdm(range(T), desc='Generating noise'):
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

            # Flatten and write to memory-mapped file
            flat_noise = np.concatenate([p.numpy().ravel() for p in step_noise])
            Z_mm[step_idx] = flat_noise
            Z_mm.flush()

            # Clean up
            del step_noise, flat_noise
        del Z_mm
        gc.collect()
        print("Step 1: Raw Noise generated and saved to disk")

        # Step 5: Save B_matrix as .npy (already on CPU)
        
        B_np = self.B_matrix.numpy()
        
        np.save(B_file, B_np)
        # np.save(Z_file, Z_mm)
        del self.B_matrix
        
        gc.collect()
        torch.cuda.empty_cache()

        print("Step 2: B_matrix saved to disk")

        # Step 6: Run GPU matrix multiplication
        gpu_matrix_multiply(
            T=T,
            D=D,
            a_path=B_file,
            b_path=Z_file,
            out_path=result_file,
            # tile_size=1024
        )

        print("Step 3: Matrix multiplication completed")


    def add_noise(self):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """
        
        # If no noise history exists, generate it now
        device = next(iter(self.params)).device
        T = self.num_steps
        D = sum(p.numel() for p in self.params) 
        if self.correlated_noise_history is None:
            print('Generating Noise start ...')
            print('Pre-generating all the noise -- It might take some time but next steps will not')
            self._generate_all_noise()
            self.correlated_noise_history = 1
            print('generation is done')

        # Get the noise for the current step
        flat_tensor = self.get_correlated_step(self.current_step,T,D)
        step__noise_correlated = self.reshape_flat_to_params(flat_tensor, self.params)
        # step__noise_correlated = self.correlated_noise_history[self.current_step] #BZ
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