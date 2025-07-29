from typing import Optional
import torch

class LagrangeTerms:
  def __init__(
      self,
      lagrange_multiplier: Optional[torch.Tensor] = None,
      contrib_matrix: Optional[torch.Tensor] = None,
      u_multipliers: Optional[torch.Tensor] = None,
      u_matrices: Optional[torch.Tensor] = None,
      nonneg_multiplier: Optional[torch.Tensor] = None
  ):
      self.lagrange_multiplier = lagrange_multiplier
      self.contrib_matrix = contrib_matrix
      self.u_multipliers = u_multipliers
      self.u_matrices = u_matrices
      self.nonneg_multiplier = nonneg_multiplier

      # Determine a common dtype and device
      for candidate in [lagrange_multiplier, u_multipliers, nonneg_multiplier, contrib_matrix]:
          if candidate is not None:
              ref = candidate
              break
      else:
          raise ValueError("Cannot infer dtype/device from None inputs")

      self.dtype = ref.dtype
      self.device = ref.device

      # Sanity check that all tensors (if not None) match dtype and device
      for name, t in {
          "lagrange_multiplier": lagrange_multiplier,
          "contrib_matrix": contrib_matrix,
          "u_multipliers": u_multipliers,
          "u_matrices": u_matrices,
          "nonneg_multiplier": nonneg_multiplier,
      }.items():
          if t is not None:
              assert t.dtype == self.dtype, f"{name} has dtype {t.dtype}, expected {self.dtype}"
              assert t.device == self.device, f"{name} is on {t.device}, expected {self.device}"
      

  @property
  def num_iters(self):
    """Returns `n`, the number of iterations."""
    n_list = []
    if self.contrib_matrix is not None:
        n_list.append(self.contrib_matrix.shape[0])
    if self.u_matrices is not None:
        n_list.append(self.u_matrices.shape[1])
    if self.nonneg_multiplier is not None:
        n_list.append(self.nonneg_multiplier.shape[0])
    if not n_list:
        raise ValueError("No way to determine num_iters")
    assert all(n == n_list[0] for n in n_list), f"Inconsistent num_iters: {n_list}"
    return n_list[0]

  def u_total(self):
    """Summarize as a single tr(U @ X) matrix in the Lagrangian."""
    n = self.num_iters
    u_total = torch.zeros((n, n), dtype=self.dtype, device=self.device)

    if self.lagrange_multiplier is not None:
      assert self.contrib_matrix is not None
      h = self.contrib_matrix
      assert self.lagrange_multiplier.shape[0] == h.shape[1]
      u_total += h @ torch.diag(self.lagrange_multiplier) @ h.T

    if self.u_matrices is not None:
      assert self.u_multipliers is not None
      u_total += torch.einsum("i,ijk->jk", self.u_multipliers, self.u_matrices)

    if self.nonneg_multiplier is not None:
      u_total -= self.nonneg_multiplier
    return u_total
    
  def assert_valid(self):
    _ = self.num_iters  # Basic shape checks
    assert (self.lagrange_multiplier is not None) or (
        self.u_matrices is not None
    )
    if self.lagrange_multiplier is not None:
      assert torch.all(self.lagrange_multiplier >= 0), self.lagrange_multiplier
    if self.u_multipliers is not None:
      assert torch.all(self.u_multipliers >= 0.0)
    if self.nonneg_multiplier is not None:
      assert torch.all(self.nonneg_multiplier >= 0.0), torch.min(
          self.nonneg_multiplier
      )

  def multiplier_sum(self):
    s = 0.0
    if self.lagrange_multiplier is not None:
        s += torch.sum(self.lagrange_multiplier).item()
    if self.u_multipliers is not None:
        s += torch.sum(self.u_multipliers).item()
    assert s > 0.0
    return s

  def replace(self, **kwargs):
    """Return a new LagrangeTerms object with some fields replaced."""
    updated = {
        "lagrange_multiplier": self.lagrange_multiplier,
        "contrib_matrix": self.contrib_matrix,
        "u_multipliers": self.u_multipliers,
        "u_matrices": self.u_matrices,
        "nonneg_multiplier": self.nonneg_multiplier,
    }
    updated.update(kwargs)
    return LagrangeTerms(**updated)

def summarize(lt: LagrangeTerms) -> LagrangeTerms:
  """Summarizes vector terms with one new matrix term."""
  assert lt.lagrange_multiplier is not None, 'No per-vector multipliers to summarize'

  n = lt.num_iters
  multiplier_sum = torch.sum(lt.lagrange_multiplier).item()

  u_matrix = (
      lt.contrib_matrix
      @ torch.diag(lt.lagrange_multiplier)
      @ lt.contrib_matrix.T
  ) / multiplier_sum

  u_matrices = u_matrix.reshape(1, n, n)
  u_multipliers = torch.tensor([multiplier_sum], dtype=lt.dtype, device=lt.device)

  if lt.u_matrices is not None:
      assert lt.u_multipliers is not None
      u_matrices = torch.cat([lt.u_matrices, u_matrices], dim=0)
      u_multipliers = torch.cat([lt.u_multipliers, u_multipliers], dim=0)

  return LagrangeTerms(
      lagrange_multiplier=None,
      contrib_matrix=None,
      u_multipliers=u_multipliers,
      u_matrices=u_matrices,
      nonneg_multiplier=lt.nonneg_multiplier,
  )


