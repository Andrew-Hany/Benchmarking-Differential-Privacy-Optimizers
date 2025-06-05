import torch
from typing import Optional, Tuple, List
import time
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def diagonalize_and_take_torch_matrix_sqrt(matrix: torch.Tensor, min_eval: float = 0.0) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(matrix)
    eval_sqrt = torch.clamp(evals, min=min_eval).sqrt()
    return evecs @ torch.diag(eval_sqrt) @ evecs.T

def hermitian_adjoint(matrix: torch.Tensor) -> torch.Tensor:
    return matrix.conj().T

def compute_loss_in_x(target: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Loss proxy: trace(Aᵀ A X⁻¹) * max(diag(X))"""
    m = hermitian_adjoint(target) @ target @ torch.linalg.inv(x)
    raw_trace = torch.trace(m)
    max_diag = torch.max(torch.diag(x))
    return raw_trace * max_diag

def compute_normalized_x_from_vector(matrix: torch.Tensor, v: torch.Tensor, precomputed_sqrt: Optional[torch.Tensor] = None) -> torch.Tensor:
    inv_diag_sqrt = torch.diag(v.pow(-0.5))
    diag_sqrt = torch.diag(v.pow(0.5))
    if precomputed_sqrt is None:
        target = hermitian_adjoint(matrix) @ matrix
        matrix_sqrt = diagonalize_and_take_torch_matrix_sqrt(diag_sqrt @ target @ diag_sqrt)
    else:
        matrix_sqrt = precomputed_sqrt
    x = inv_diag_sqrt @ matrix_sqrt @ inv_diag_sqrt
    x_diag = torch.diag(x)
    normalized_x = torch.diag(x_diag.pow(-0.5)) @ x @ torch.diag(x_diag.pow(-0.5))
    return normalized_x

def compute_phi_fixed_point(
    matrix: torch.Tensor,
    initial_v: torch.Tensor,
    rtol: float = 1e-5,
    max_iterations: Optional[int] = None,
) -> Tuple[torch.Tensor, int, float, List[float], List[float]]:

    matrix = matrix.to(device)
    initial_v = initial_v.to(device)

    target = hermitian_adjoint(matrix) @ matrix
    v = initial_v.clone()
    n_iters = 0
    time_array = []
    loss_array = []

    def continue_loop(iteration: int) -> bool:
        if max_iterations is None:
            return True
        return iteration < max_iterations

    def _compute_loss(v: torch.Tensor, matrix_sqrt: torch.Tensor) -> torch.Tensor:
        normalized_x = compute_normalized_x_from_vector(matrix, v, matrix_sqrt)
        return compute_loss_in_x(matrix, normalized_x)

    def _update_loss(v: torch.Tensor, matrix_sqrt: torch.Tensor):
        time_array.append(time.time() - start)
        loss = _compute_loss(v, matrix_sqrt)
        loss_array.append(loss.item())

    matrix_sqrt = diagonalize_and_take_torch_matrix_sqrt(torch.diag(v.sqrt()) @ target @ torch.diag(v.sqrt()))
    start = time.time()

    while continue_loop(n_iters):
        _update_loss(v, matrix_sqrt)

        new_v = torch.diag(matrix_sqrt)
        matrix_sqrt = diagonalize_and_take_torch_matrix_sqrt(torch.diag(new_v.sqrt()) @ target @ torch.diag(new_v.sqrt()))

        norm_diff = torch.linalg.norm(new_v - v)
        rel_norm_diff = norm_diff / torch.linalg.norm(v)

        if rel_norm_diff < rtol:
            _update_loss(new_v, matrix_sqrt)
            # print(n_iters+1)
            break

        v = new_v
        n_iters += 1
    # print(n_iters)
    return v, n_iters, rel_norm_diff.item(), time_array, loss_array

def optimize_factorization_fixed_point(
    s_matrix: torch.Tensor,
    max_iterations: int = 100,
    rtol: float = 1e-5,
    initial_v: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, List[float], List[float]]:

    s_matrix = s_matrix.to(device)
    n = s_matrix.shape[0]

    if initial_v is None:
        initial_v = torch.ones(n, dtype=s_matrix.dtype, device=device)

    v, n_iters, final_relnorm, timing, losses = compute_phi_fixed_point(
        s_matrix, initial_v, rtol, max_iterations
    )

    

    initial_time = timing[0] if timing else 0.0
    adj_time_array = [t - initial_time for t in timing]

    X_normalized = compute_normalized_x_from_vector(s_matrix,v)

    return X_normalized, losses, adj_time_array




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
def compute_sensitivity(C: torch.Tensor) -> float:
    if C is None:
        return 1.0
    col_norms = torch.linalg.norm(C, dim=0)  # ℓ2 norm of each column
    return torch.max(col_norms).item()
    
# n=500
# S_torch = torch.tril(torch.ones((n, n), dtype=torch.float64))
# x_torch, _, _ = optimize_factorization_fixed_point(S_torch, max_iterations=100,rtol=1e-10)
# B,C = B_and_C_from_x_and_s(x_torch,S_torch)

# print(B)
# print(C)
# print(compute_sensitivity(C))

