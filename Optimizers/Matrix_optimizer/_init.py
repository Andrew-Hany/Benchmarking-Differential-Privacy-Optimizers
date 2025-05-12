
from .MatrixSGD2 import DPOptimizer_Matrix
import torch 

def get_optimizer_class():
        return DPOptimizer_Matrix

# we need to do the offline factorizatoin somewhere here
# we can Then use the Factorization function to get the required matrices B and C 
# Then we can use them in privacy engine method _prepare_optimizer: which them call the get optimizer class

def compute_sensitivity(C):
    if C is None:
        return 1.0
    return torch.linalg.norm(C, ord=2).item()