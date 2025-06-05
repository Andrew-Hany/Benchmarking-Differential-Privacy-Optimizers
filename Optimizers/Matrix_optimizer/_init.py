
from .MatrixSGD2 import DPOptimizer_Matrix
from .matrix_factorization_pytorch_fixed_point import compute_sensitivity, optimize_factorization_fixed_point, B_and_C_from_x_and_s
from .lambda_construction import build_lambda_tensor
import torch 

def get_optimizer_class():
        return DPOptimizer_Matrix

# we need to do the offline factorizatoin somewhere here
# we can Then use the Factorization function to get the required matrices B and C 
# Then we can use them in privacy engine method _prepare_optimizer: which them call the get optimizer class

def get_matrix_B_and_C_single_epoch_fixed_point(n, lamda_matrix=False):
    if lamda_matrix == True:
        T,tau = n, n
        Lambda = build_lambda_tensor(T,tau)
        S_torch = torch.tril(torch.ones((n, n), dtype=torch.float64))
        S_torch  = Lambda @ S_torch    
    else:
        S_torch = torch.tril(torch.ones((n, n), dtype=torch.float64))

    x_torch, _, _ = optimize_factorization_fixed_point(S_torch, max_iterations=100,rtol=1e-10)
    B,C = B_and_C_from_x_and_s(x_torch,S_torch)
    return B,C
