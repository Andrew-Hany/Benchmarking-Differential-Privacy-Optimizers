import torch
from typing import Tuple


def _make_permutation_matrix(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Constructs a matrix with 1s on the anti-diagonal and 0s elsewhere."""
    perm = torch.zeros((n, n), dtype=dtype, device=device)
    for i in range(n):
        perm[i, n - i - 1] = 1.0
    return perm

def _permute_lower_triangle(matrix: torch.Tensor) -> torch.Tensor:
    """Computes P @ matrix @ P^T using anti-diagonal permutation matrix."""
    P = _make_permutation_matrix(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    return P @ matrix @ P.T

def B_and_C_from_x_and_s(x_matrix: torch.Tensor, s_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns W and H such that X = H^T H and S ≈ W H."""
    device = s_matrix.device
    dtype = s_matrix.dtype

    # Step 1: Cholesky on time-reversed X
    x_perm = _permute_lower_triangle(x_matrix)
    h_lower = torch.linalg.cholesky(x_perm)

    # Step 2: Reverse permutation to get original orientation
    h_matrix = _permute_lower_triangle(h_lower.T).to(dtype=dtype, device=device)
    # h_matrix = h_matrix / torch.maximum(col_norms, torch.ones_like(col_norms))

    # Step 3: Compute B = S @ H^{-1}
    h_inv = torch.linalg.inv(h_matrix)
    w_matrix = s_matrix @ h_inv

    return w_matrix, h_matrix