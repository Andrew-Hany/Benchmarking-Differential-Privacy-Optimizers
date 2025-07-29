
from .MatrixSGD import DPOptimizer_Matrix
from .matrix_factorization_pytorch_fixed_point import compute_sensitivity, optimize_factorization_fixed_point, B_and_C_from_x_and_s
from .lambda_construction import build_lambda_tensor
import torch 


from .multiple_participations_Pytorch.optimization import solve_lagrange_dual_problem, DualUpdate
from .multiple_participations_Pytorch.lagrange_terms import LagrangeTerms
from .multiple_participations_Pytorch.lt_initializers import init_nonnegative_lagrange_terms
from .multiple_participations_Pytorch.construct_bc import B_and_C_from_x_and_s
def get_optimizer_class():
        return DPOptimizer_Matrix

# we need to do the offline factorizatoin somewhere here
# we can Then use the Factorization function to get the required matrices B and C 
# Then we can use them in privacy engine method _prepare_optimizer: which them call the get optimizer class

def get_matrix_B_and_C_single_epoch_fixed_point(n, lamda_matrix=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if lamda_matrix == True:
        T,tau = n, n
        Lambda = build_lambda_tensor(T,tau)
        S_torch = torch.tril(torch.ones((n, n), dtype=torch.float64, device = device))
        Lambda = Lambda.to(device)
        S_torch  = Lambda @ S_torch    
    else:
        S_torch = torch.tril(torch.ones((n, n), dtype=torch.float64, device = device))

    x_torch, _, _ = optimize_factorization_fixed_point(S_torch, max_iterations=1000,rtol=1e-10)
    B,C = B_and_C_from_x_and_s(x_torch,S_torch)
    return B,C

def get_matrix_B_and_C_multi_epoch(num_epochs, steps_per_epoch, lamda_matrix=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = num_epochs*steps_per_epoch
    if lamda_matrix == True:
        T,tau = n, n
        Lambda = build_lambda_tensor(T,tau)
        S = torch.tril(torch.ones((n, n), dtype=torch.float64, device = device))
        Lambda = Lambda.to(device)
        S  = Lambda @ S 
    else:
        S = torch.tril(torch.ones((n, n), dtype=torch.float64, device = device))

    lt = init_nonnegative_lagrange_terms(num_epochs, steps_per_epoch)

    optimizer = torch.optim.SGD([torch.nn.Parameter(lt.nonneg_multiplier.clone(), requires_grad=True)], lr=0.025, momentum=0.95 ) # for 30 epochs amd 49 steps
    # optimizer = torch.optim.SGD([torch.nn.Parameter(lt.nonneg_multiplier.clone(), requires_grad=True)], lr=0.04, momentum=0.95 ) # for 25 epochs amd 59 steps
    # optimizer = torch.optim.SGD([torch.nn.Parameter(lt.nonneg_multiplier.clone(), requires_grad=True)], lr=0.0452, momentum=0.96 ) # for 20 epochs amd 59 steps
    # optimizer = torch.optim.SGD([torch.nn.Parameter(lt.nonneg_multiplier.clone(), requires_grad=True)], lr=0.043, momentum=0.96 ) # for 20 epochs amd 49 steps
    update_fn = DualUpdate(nonneg_optimizer=optimizer, lt=lt, multiplicative_update=True)


    results = solve_lagrange_dual_problem(
        s_matrix=S,
        lt=lt,
        update_lagrange_terms_fn=update_fn,
        max_iterations=15_000,
        iters_per_eval=100,
        target_relative_duality_gap=0.01,
        # verbose=False
    )
    return results["B"],results["C"]
