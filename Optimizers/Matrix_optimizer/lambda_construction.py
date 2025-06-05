import torch
import math
import matplotlib.pyplot as plt

def build_lambda_tensor(T: int, tau: int) -> torch.Tensor:
    """
    Construct Λ_τ (Lambda_tau) using PyTorch.

    Parameters
    ----------
    T : int
        Total dimension (matrix will be T x T).
    tau : int
        Restart interval τ.

    Returns
    -------
    torch.Tensor
        Tensor of shape (T, T) representing Λ_τ.
    """
    lam = torch.zeros((T, T), dtype=torch.float64)
    sqrt_tau_inv = 1.0 / math.sqrt(tau)

    for row in range(T):
        t = row + 1
        if t % tau != 0:
            lam[row, row] = sqrt_tau_inv
            base = (t // tau) * tau
            if t > tau:
                lam[row, base - 1] = -sqrt_tau_inv
        else:
            lam[row, row] = 1.0
            prev = t - tau
            if t > tau:
                lam[row, prev - 1] = -1.0
    return lam

# # Example:
# T, tau = 12, 3
# Lambda_torch = build_lambda_tensor(T, tau)

# # heat-map with diverging palette centred on 0
# plt.figure(figsize=(5, 4))
# plt.imshow(Lambda_torch, cmap="seismic", vmin=-1, vmax=1, aspect="auto")
# plt.colorbar(label="entry value")
# plt.title(f"Λτ with diverging palette (T={T}, τ={tau})")
# plt.xlabel("j")
# plt.ylabel("t")
# plt.tight_layout()
# plt.show()


