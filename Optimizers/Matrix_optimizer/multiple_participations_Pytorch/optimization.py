# from multi_epoch_dp_matrix_factorization import loops
# from multi_epoch_dp_matrix_factorization.multiple_participations import lagrange_terms


_MIN_INNER_EIGENVALUE = 1e-20

import torch
from torch import optim
from typing import Tuple, Optional
import time
from collections import OrderedDict

from .lagrange_terms import LagrangeTerms 
from .construct_bc import B_and_C_from_x_and_s

def sqrt_and_sqrt_inv(
    matrix: torch.Tensor,
    min_eigenval: float = 0.0,
    compute_inverse: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Matrix square root and its inverse for symmetric PSD matrices."""
    # Eigen decomposition (assumes symmetric input)
    evals, evecs = torch.linalg.eigh(matrix)

    # Clamp eigenvalues for stability
    eval_sqrt = torch.clamp(evals, min=min_eigenval).sqrt()

    # Matrix square root
    sqrt = evecs @ torch.diag(eval_sqrt) @ evecs.T

    # Matrix inverse square root (if requested)
    sqrt_inv = None
    if compute_inverse:
        sqrt_inv = evecs @ torch.diag(1.0 / eval_sqrt) @ evecs.T

    return sqrt, sqrt_inv

def x_and_x_inv_from_dual(
    lt: LagrangeTerms,
    target: torch.Tensor,
    compute_inv: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Computes matrix X (and optionally X^{-1}) that minimizes the Lagrangian
    given dual variables in lt and sensitivity matrix `target`.

    Args:
        lt: LagrangeTerms object
        target: torch.Tensor (usually S^T S)
        compute_inv: Whether to return X^{-1} as well

    Returns:
        X, and optionally X^{-1}
    """
    u_total = lt.u_total()  # (n x n), should be symmetric PSD
    
    # Step 1: diag_sqrt and its inverse
    diag_sqrt, inv_diag_sqrt = sqrt_and_sqrt_inv(u_total, _MIN_INNER_EIGENVALUE, compute_inverse=True)

    # Step 2: Inner term = diag_sqrt @ target @ diag_sqrt
    inner_term = diag_sqrt @ target @ diag_sqrt

    # Step 3: sqrt of inner term and its inverse
    sqrt_inner, inv_sqrt_inner = sqrt_and_sqrt_inv(inner_term, _MIN_INNER_EIGENVALUE, compute_inverse=compute_inv)

    # Step 4: Final X and optional X⁻¹
    x_matrix = inv_diag_sqrt @ sqrt_inner @ inv_diag_sqrt

    if compute_inv:
        x_inv = diag_sqrt @ inv_sqrt_inner @ diag_sqrt
    else:
        x_inv = None

    return x_matrix, x_inv

class DualUpdate:
  """Equivalent to OptaxUpdate, using PyTorch's optimizer interface."""

  def __init__(
      self,
      nonneg_optimizer: Optional[torch.optim.Optimizer] = None,
      lt: Optional[LagrangeTerms] = None,
      multiplicative_update: bool = False,
  ):
    self.multiplicative_update = multiplicative_update
    self.nonneg_opt_state = None
    self.nonneg_optimizer = None
    if nonneg_optimizer is not None and lt is not None and lt.nonneg_multiplier is not None:
        # Equivalent of: nonneg_opt_state = nonneg_optimizer.init(...)
        # PyTorch expects actual parameters, so we wrap nonneg_multiplier
        self.nonneg_multiplier_param = torch.nn.Parameter(
            lt.nonneg_multiplier.clone(), requires_grad=True
        )

        # Reinitialize the optimizer with this parameter (equivalent to .init in JAX)
        # equivalent to torch.optim.SGD([param], lr=0.01, momentum=0.9)
        self.nonneg_optimizer = type(nonneg_optimizer)([self.nonneg_multiplier_param], **nonneg_optimizer.defaults)

        # Optional: store state dict if needed (like nonneg_opt_state)
        self.nonneg_opt_state = self.nonneg_optimizer.state_dict()
    else:
        self.nonneg_multiplier_param = None


  def __call__(self, lt: LagrangeTerms, target: torch.Tensor):
    x_matrix, _ = x_and_x_inv_from_dual(lt, target, compute_inv=False)
    lt_updates = {}

    # --- Update lagrange_multiplier ---
    if lt.lagrange_multiplier is not None:
        mult_update = torch.diag(lt.contrib_matrix.T @ x_matrix @ lt.contrib_matrix)
        assert torch.all(mult_update >= 0), f"x_matrix may not be PD, mult_update:\n {mult_update}"
        lt_updates["lagrange_multiplier"] = lt.lagrange_multiplier * mult_update
        assert torch.all(lt_updates["lagrange_multiplier"] > 0)

    # --- Update u_multipliers (in-place) ---
    if lt.u_matrices is not None:
        assert len(lt.u_multipliers) == len(lt.u_matrices)
        for i, u_matrix in enumerate(lt.u_matrices):
            lt.u_multipliers[i] = lt.u_multipliers[i] * torch.trace(u_matrix @ x_matrix)

    # --- Update nonneg_multiplier ---
    if lt.nonneg_multiplier is not None:
        grads = x_matrix.clone()

        if self.multiplicative_update:
            grads *= lt.nonneg_multiplier

        if self.nonneg_optimizer is not None and self.nonneg_multiplier_param is not None:
            self.nonneg_multiplier_param.data = lt.nonneg_multiplier.clone().detach().requires_grad_()

            self.nonneg_optimizer.zero_grad()
            self.nonneg_multiplier_param.grad = grads
            self.nonneg_optimizer.step()

            updated = self.nonneg_multiplier_param.data
        else:
            raise RuntimeError("No optimizer provided for nonneg_multiplier, cannot update.")
        # Clamp to enforce non-negativity
        lt_updates["nonneg_multiplier"] = torch.clamp(updated, min=0.0)

    return lt.replace(**lt_updates)

def lagrangian_fn(x_matrix: torch.Tensor, lt: LagrangeTerms) -> torch.Tensor:
    """Evaluates the Lagrangian function, where x_matrix corresponds to lt."""
    return 2 * torch.trace(lt.u_total() @ x_matrix) - lt.multiplier_sum()

def max_min_sensitivity_squared_for_x(
    x_matrix: torch.Tensor, lt: LagrangeTerms
) -> tuple[float, float]:
    """Returns (max, min) sensitivity**2 based on lt constraints and x_matrix."""
    sens_list = []

    if lt.contrib_matrix is not None:
        h = lt.contrib_matrix
        s = torch.diagonal(h.T @ x_matrix @ h)
        max_s, min_s = torch.max(s), torch.min(s)
        assert min_s >= -1e-10, f"min_s = {min_s.item()}"
        sens_list.extend([max_s.item(), min_s.item()])

    if lt.u_matrices is not None:
        traces = [torch.trace(u @ x_matrix).item() for u in lt.u_matrices]
        sens_list.extend(traces)

    assert sens_list, "sensitivity list is empty"
    return max(sens_list), min(sens_list)


def solve_lagrange_dual_problem(
    s_matrix: torch.Tensor,
    lt: LagrangeTerms,
    target_relative_duality_gap: float = 0.001,
    update_lagrange_terms_fn=None,
    max_iterations: int = 1000000,
    iters_per_eval: int = 1,
    verbose: bool = True
):
    if update_lagrange_terms_fn is None:
        optimizer = torch.optim.SGD([torch.nn.Parameter(lt.nonneg_multiplier.clone(), requires_grad=True)], lr=0.01, momentum=0.95)
        update_lagrange_terms_fn = DualUpdate(
            nonneg_optimizer=optimizer,
            lt=lt,
            multiplicative_update=False,
        )

    target = s_matrix.T @ s_matrix
    start_time = time.time()
    losses = []
    dual_obj_vals = []
    num_iters = 0

    while True:
        for _ in range(iters_per_eval):
            lt = update_lagrange_terms_fn(lt, target)
            lt.assert_valid()
            num_iters += 1

        x_matrix, _ = x_and_x_inv_from_dual(lt, target, compute_inv=False)
        feasible_x_matrix = x_matrix.clone()

        if lt.nonneg_multiplier is not None:
            feasible_x_matrix = torch.clamp(feasible_x_matrix, min=0.0)

        max_sens_sq, min_sens_sq = max_min_sensitivity_squared_for_x(feasible_x_matrix, lt)
        
        feasible_x_matrix /= max(max_sens_sq, 1.0)
        feasible_x_inv = torch.linalg.inv(feasible_x_matrix)

        primal_obj = torch.trace(target @ feasible_x_inv).item()
        dual_obj = lagrangian_fn(x_matrix, lt).item()
        duality_gap = primal_obj - dual_obj
        relative_gap = duality_gap / primal_obj

        losses.append(primal_obj)
        dual_obj_vals.append(dual_obj)

        if verbose:
            log_str = (
                f"{num_iters:5d}  primal obj {primal_obj:8.2f} dual obj"
                f" {dual_obj:8.2f} gap {duality_gap:8.2f} relative {relative_gap:7.2g},"
                f" min(x)={x_matrix.min():8.6f}, "
                f"max v {lt.lagrange_multiplier.max() if lt.lagrange_multiplier is not None else 'N/A'}, "
                f"max nonneg v {lt.nonneg_multiplier.max() if lt.nonneg_multiplier is not None else 'N/A'}, "
                f"max/min sens^2 {max_sens_sq:.6f}/{min_sens_sq:.6f}, "
                f"elapsed {time.time() - start_time:.1f}s"
            )
            print(log_str)

        assert duality_gap >= -1e-10, (
            f"duality_gap {duality_gap}, dual_obj {dual_obj}, primal_obj {primal_obj}"
        )
        
        if lt.nonneg_multiplier is not None:
            assert torch.all(feasible_x_matrix >= 0), torch.min(feasible_x_matrix)
        if relative_gap <= target_relative_duality_gap:
            break
        if num_iters >= max_iterations:
            break

    # Skip W, H generation for now

    B_matrix, C_matrix = B_and_C_from_x_and_s(feasible_x_matrix, s_matrix)
    return OrderedDict(
        B=B_matrix,
        C=C_matrix,
        U=lt.u_total(),
        losses=losses,
        dual_obj_vals=dual_obj_vals,
        n_iters=num_iters,
        lagrange_terms=lt,
        x_matrix=x_matrix,
        relative_duality_gap=relative_gap,
    )