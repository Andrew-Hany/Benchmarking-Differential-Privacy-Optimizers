import numpy as np
import torch
from .contrib_matrix_builders import epoch_participation_matrix_all_positive
from .lagrange_terms import LagrangeTerms



def init_nonnegative_lagrange_terms(num_epochs, steps_per_epoch) -> LagrangeTerms:
    """Returns a dual-feasible initialization of the LagrangeTerms."""
    n = num_epochs * steps_per_epoch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Generate H as NumPy array
    epoch_contrib_matrix_np = epoch_participation_matrix_all_positive(n, num_epochs)
    # Convert to torch.Tensor
    H = torch.tensor(epoch_contrib_matrix_np, dtype=torch.float64,device=device)
    # Initialize lagrange multipliers (v_i = 1)
    v = torch.ones(H.shape[1], dtype=torch.float64, device=device)
    # Compute nonnegativity multiplier matrix: H H^T - I
    nonneg = H @ H.T - torch.eye(H.shape[0], dtype=torch.float64, device=device)
    
    # We need to carefully initialize the LagrangeTerms so that the initial
    # u_total() is PD. Further, if using a multiplicative update for the
    # the nonnegative weights (which seems to work well), we need to be
    # completely sure we only assign zeros where we know the optimal multiplier
    # for the non-negativity constraint is zero. Fortunately, the structure of the
    # epoch participation structure means we can do this: we only need to worry
    # about negativity on X[i, j] when i, j are separated by exactly
    # steps_per_epoch, that is, if the same user might participate on both
    # steps i and j. It appears these entries are always zero in the optimal X,
    # with the non-negativity constraint being tight.
    #
    # This results in the initial u_total() being the identity matrix.
    return LagrangeTerms(  # pytype: disable=wrong-arg-types  # jnp-array
        lagrange_multiplier=v,
        contrib_matrix=H,
        nonneg_multiplier=nonneg
    )
