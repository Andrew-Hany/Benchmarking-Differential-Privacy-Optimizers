
from .MatrixSGD import DPOptimizer_Matrix


def get_optimizer_class(
    clipping: str, distributed: bool, grad_sample_mode: str = None, Dice: bool = False
):
        return DPOptimizer_Matrix

# we need to do the offline factorizatoin somewhere here
# we can Then use the Factorization function to get the required matrices B and C 
# Then we can use them in privacy engine method _prepare_optimizer: which them call the get optimizer class

