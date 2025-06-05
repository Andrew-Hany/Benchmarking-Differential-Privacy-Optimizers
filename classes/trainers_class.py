import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import warnings
import opacus
import sys
import os


from abc import ABC, abstractmethod
from Optimizers.Adam_optimizer.AdamBC import *
from privacy_engines.Dice_privacy_engine import Dice_PrivacyEngine
from privacy_engines.KFprivacy_engine import KF_PrivacyEngine
from privacy_engines.Matrix_single_epoch_privacy_engine import  Matrix_single_epoch_PrivacyEngine
from privacy_engines.Matrix_single_epoch_lambda_privacy_engine import Matrix_single_epoch_lambda_PrivacyEngine
from .training_loops import TrainingOrchestrator
from opacus.accountants.utils import get_noise_multiplier
import opacus

from opacus.accountants.rdp import RDPAccountant
# Define extended alphas
extended_alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)) +   [ 70, 80, 90, 100,200, 300, 400, 512,1024,2048,3000]
# Overwrite the default list used by RDPAccountant
RDPAccountant.DEFAULT_ALPHAS = extended_alphas


# Registry for optimizer-specific classes
OptimizerRegistry = {}

def register_optimizer(optimizer_type):
    """
    Decorator to register an optimizer-specific class for a specific optimizer type.
    Args:
        optimizer_type: The type of optimizer (e.g., "SGD", "DICE").
    Returns:
        A decorator that registers the optimizer-specific class.
    """
    def decorator(cls):
        OptimizerRegistry[optimizer_type.upper()] = cls
        return cls
    return decorator
class BaseOptimizer_trainer(ABC):
    @abstractmethod
    def train(
        self,
        model: torch.nn.Module,
        model_type,
        train_loader: torch.utils.data.DataLoader,
        test_loader:torch.utils.data.DataLoader,
        learning_rate,
        sample_rate,
        criterion,
        num_epochs: int,
        target_epsilon: float,
        clip_bound: float,
        delta: float,
        device,
        accountant='prv',
        normalize_clipping=False,
        random_seed=474237,
        verbose=False,
        **kwargs
    ):
        """
        Abstract method to define the training process for an optimizer.
        Args:
            model: PyTorch model.
            model_type: string
            train_loader: DataLoader for training data.
            learning_rate: Learning rate for the optimizer.
            sample_rate: Sampling rate for the dataset.
            criterion: Loss function.
            num_epochs: Number of training epochs.
            target_epsilon: Target epsilon for differential privacy.
            clip_bound: Gradient clipping bound.
            delta: Delta for differential privacy.
            device: Device (CPU/GPU).
            normalize_clipping: Whether to normalize gradient clipping.
            random_seed: Random seed for reproducibility.
            verbose: Whether to print progress.
            **kwargs: Additional parameters specific to the optimizer.
        Returns:
            Tuple of (epsilon, noise_multiplier, all_losses, all_accuracies, elapsed_time).
        """
        pass

@register_optimizer("SGD")
class DP_SGD_train_epsilon(BaseOptimizer_trainer):
    @staticmethod
    def train(
        model: torch.nn.Module,
        model_type,
        train_loader: torch.utils.data.DataLoader,
        test_loader:torch.utils.data.DataLoader,
        learning_rate,
        sample_rate,
        criterion,
        num_epochs: int,
        target_epsilon: float,
        clip_bound: float,
        delta: float,
        device,
        accountant='prv',
        normalize_clipping= False,
        random_seed=474237,
        verbose=False,
        **kwargs):
        privacy_engine = opacus.PrivacyEngine(
            accountant=accountant,
            secure_mode=False,  # Should be set to True for production use
        )

        noise_multiplier = get_noise_multiplier(
                    target_epsilon=target_epsilon,
                    target_delta=delta,
                    sample_rate=sample_rate,
                    epochs=num_epochs,
                    accountant=privacy_engine.accountant.mechanism(),
        )
        optimizer = optim.SGD(model.parameters(), learning_rate)
        rng = torch.Generator(device=device)
        rng.manual_seed(int(random_seed))

        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=target_epsilon,
            target_delta=delta,
            epochs=num_epochs,
            max_grad_norm=clip_bound,
            noise_generator=rng,
            loss_reduction="mean",
            normalize_clipping = normalize_clipping,
        )

        orchestrator = TrainingOrchestrator()
        all_train_losses, all_train_accuracies,all_test_losses,all_test_accuracies,elapsed_time = orchestrator.training_loop(model_type,num_epochs,train_loader,test_loader,model,criterion,optimizer,device,verbose=verbose)
        epsilon = privacy_engine.get_epsilon(delta)
        return epsilon,noise_multiplier, all_train_losses, all_train_accuracies,all_test_losses,all_test_accuracies,elapsed_time

@register_optimizer("DICE")
class DP_DICE_train_epsilon(BaseOptimizer_trainer):
    @staticmethod
    def train(
        model: torch.nn.Module,
        model_type,
        train_loader: torch.utils.data.DataLoader,
        test_loader:torch.utils.data.DataLoader,
        learning_rate,
        sample_rate,
        criterion,
        num_epochs: int,
        target_epsilon: float,
        clip_bound: float,
        delta: float,
        device,
        accountant='prv',
        normalize_clipping= False,
        random_seed=474237,
        verbose=False,
        **kwargs):


        optimizer = optim.SGD(model.parameters(), learning_rate)
        privacy_engine = Dice_PrivacyEngine(
            accountant=accountant,
            # accountant='rdp',
            secure_mode=False,
        )
        
        noise_multiplier = get_noise_multiplier(
                    target_epsilon=target_epsilon,
                    target_delta=delta,
                    sample_rate=sample_rate,
                    epochs=num_epochs,
                    accountant=privacy_engine.accountant.mechanism(),
        )

        rng = torch.Generator(device=device)
        rng.manual_seed(int(random_seed))

        # making the model and optimizer private
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            target_delta = delta,
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=target_epsilon,
            epochs=num_epochs,
            max_grad_norm=clip_bound,
            noise_generator=rng,
            loss_reduction="mean",
            normalize_clipping = normalize_clipping,
            error_max_grad_norm = kwargs['error_max_grad_norm']
        )



        # Training Loop
        orchestrator = TrainingOrchestrator()
        all_train_losses, all_train_accuracies,all_test_losses,all_test_accuracies,elapsed_time = Training.training_loop(model_type,num_epochs,train_loader,test_loader,model,criterion,optimizer,device,verbose=verbose)
        epsilon = privacy_engine.get_epsilon(delta)
        return epsilon,noise_multiplier, all_train_losses, all_train_accuracies,all_test_losses,all_test_accuracies,elapsed_time

@register_optimizer("ADAMBC")
class DP_ADAM_train_epsilon(BaseOptimizer_trainer):
    @staticmethod
    def train(
        model: torch.nn.Module,
        model_type,
        train_loader: torch.utils.data.DataLoader,
        test_loader:torch.utils.data.DataLoader,
        learning_rate,
        sample_rate,
        criterion,
        num_epochs: int,
        target_epsilon: float,
        clip_bound: float,
        delta: float,
        device,
        accountant='prv',
        normalize_clipping= False,
        random_seed=474237,
        verbose=False,
        **kwargs):



        privacy_engine = opacus.PrivacyEngine(
            accountant=accountant,
            # accountant='rdp',
            secure_mode=False,  # Should be set to True for production use
        )
        noise_multiplier = get_noise_multiplier(
                    target_epsilon=target_epsilon,
                    target_delta=delta,
                    sample_rate=sample_rate,
                    epochs=num_epochs,
                    accountant=privacy_engine.accountant.mechanism(),
                    # Add any additional keyword arguments if needed
                )

        # print(noise_multiplier)
        optimizer=AdamCorr(
            model.parameters(), lr=learning_rate,
            dp_batch_size=int(len(train_loader.dataset) * sample_rate),
            dp_l2_norm_clip=clip_bound,dp_noise_multiplier=noise_multiplier
        )
        rng = torch.Generator(device=device)
        rng.manual_seed(int(random_seed))

        # making the model and optimizer private
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            target_delta = delta,
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=target_epsilon,
            epochs=num_epochs,
            max_grad_norm=clip_bound,
            noise_generator=rng,
            loss_reduction="mean",
            normalize_clipping = normalize_clipping,
        )


        # Training Loop
        orchestrator = TrainingOrchestrator()
        all_train_losses, all_train_accuracies,all_test_losses,all_test_accuracies,elapsed_time = orchestrator.training_loop(model_type,num_epochs,train_loader,test_loader,model,criterion,optimizer,device,verbose=verbose)
        epsilon = privacy_engine.get_epsilon(delta)
        return epsilon,noise_multiplier, all_train_losses, all_train_accuracies,all_test_losses,all_test_accuracies,elapsed_time
@register_optimizer("KF")
class DP_KF_train_epsilon(BaseOptimizer_trainer):
    @staticmethod
    def train(
        model: torch.nn.Module,
        model_type,
        train_loader: torch.utils.data.DataLoader,
        test_loader:torch.utils.data.DataLoader,
        learning_rate,
        sample_rate,
        criterion,
        num_epochs: int,
        target_epsilon: float,
        clip_bound: float,
        delta: float,
        device,
        accountant='prv',
        normalize_clipping= False,
        random_seed=474237,
        
        verbose=False,
        **kwargs):

        privacy_engine = KF_PrivacyEngine(
            accountant=accountant,
            # accountant='rdp',
            secure_mode=False,  # Should be set to True for production use
        )

        noise_multiplier = get_noise_multiplier(
                    target_epsilon=target_epsilon,
                    target_delta=delta,
                    sample_rate=sample_rate,
                    epochs=num_epochs,
                    accountant=privacy_engine.accountant.mechanism(),
                    # Add any additional keyword arguments if needed
        )
        # print(noise_multiplier)
        optimizer = optim.SGD(model.parameters(), learning_rate)
        rng = torch.Generator(device=device)
        rng.manual_seed(int(random_seed))

        # making the model and optimizer private
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=target_epsilon,
            target_delta = delta,
            epochs=num_epochs,
            max_grad_norm=clip_bound,
            noise_generator=rng,
            loss_reduction="mean",
            kalman=True, # need this argument
            kappa=0.7, # optional
            gamma=0.5,# optional
            normalize_clipping = normalize_clipping,

        )

        orchestrator = TrainingOrchestrator()
        all_train_losses, all_train_accuracies,all_test_losses,all_test_accuracies,elapsed_time = orchestrator.training_loop(model_type,num_epochs,train_loader,test_loader,model,criterion,optimizer,device,verbose=verbose,use_closure=True)
        epsilon = privacy_engine.get_epsilon(delta)
        return epsilon,noise_multiplier, all_train_losses, all_train_accuracies,all_test_losses,all_test_accuracies,elapsed_time

@register_optimizer("Matrix_single_epoch")
class DP_Matrix_train_epsilon(BaseOptimizer_trainer):
    @staticmethod
    def train(
        model: torch.nn.Module,
        model_type,
        train_loader: torch.utils.data.DataLoader,
        test_loader:torch.utils.data.DataLoader,
        learning_rate,
        sample_rate,
        criterion,
        num_epochs: int,
        target_epsilon: float,
        clip_bound: float,
        delta: float,
        device,
        accountant='prv',
        normalize_clipping= False,
        random_seed=474237,
        
        verbose=False,
        **kwargs):

        privacy_engine = Matrix_single_epoch_PrivacyEngine(
            accountant=accountant,
            # accountant='rdp',
            secure_mode=False,  # Should be set to True for production use
        )

        noise_multiplier = get_noise_multiplier(
                    target_epsilon=target_epsilon,
                    target_delta=delta,
                    sample_rate=sample_rate,
                    epochs=num_epochs,
                    accountant=privacy_engine.accountant.mechanism(),
        )
        optimizer = optim.SGD(model.parameters(), learning_rate)
        rng = torch.Generator(device=device)
        rng.manual_seed(int(random_seed))

        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=target_epsilon,
            target_delta=delta,
            epochs=num_epochs,
            max_grad_norm=clip_bound,
            noise_generator=rng,
            loss_reduction="mean",
            normalize_clipping = normalize_clipping,
        )

        orchestrator = TrainingOrchestrator()
        all_train_losses, all_train_accuracies,all_test_losses,all_test_accuracies,elapsed_time = orchestrator.training_loop(model_type,num_epochs,train_loader,test_loader,model,criterion,optimizer,device,verbose=verbose)
        epsilon = privacy_engine.get_epsilon(delta)
        # all_train_losses, all_train_accuracies,all_test_losses,all_test_accuracies,elapsed_time =None,None,None,None,None
        return epsilon,noise_multiplier, all_train_losses, all_train_accuracies,all_test_losses,all_test_accuracies,elapsed_time

@register_optimizer("Matrix_single_epoch_lambda")
class DP_Matrix_train_epsilon(BaseOptimizer_trainer):
    @staticmethod
    def train(
        model: torch.nn.Module,
        model_type,
        train_loader: torch.utils.data.DataLoader,
        test_loader:torch.utils.data.DataLoader,
        learning_rate,
        sample_rate,
        criterion,
        num_epochs: int,
        target_epsilon: float,
        clip_bound: float,
        delta: float,
        device,
        accountant='prv',
        normalize_clipping= False,
        random_seed=474237,
        
        verbose=False,
        **kwargs):

        privacy_engine = Matrix_single_epoch_lambda_PrivacyEngine(
            accountant=accountant,
            # accountant='rdp',
            secure_mode=False,  # Should be set to True for production use
        )

        noise_multiplier = get_noise_multiplier(
                    target_epsilon=target_epsilon,
                    target_delta=delta,
                    sample_rate=sample_rate,
                    epochs=num_epochs,
                    accountant=privacy_engine.accountant.mechanism(),
        )
        optimizer = optim.SGD(model.parameters(), learning_rate)
        rng = torch.Generator(device=device)
        rng.manual_seed(int(random_seed))

        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=target_epsilon,
            target_delta=delta,
            epochs=num_epochs,
            max_grad_norm=clip_bound,
            noise_generator=rng,
            loss_reduction="mean",
            normalize_clipping = normalize_clipping,
        )

        orchestrator = TrainingOrchestrator()
        all_train_losses, all_train_accuracies,all_test_losses,all_test_accuracies,elapsed_time = orchestrator.training_loop(model_type,num_epochs,train_loader,test_loader,model,criterion,optimizer,device,verbose=verbose)
        epsilon = privacy_engine.get_epsilon(delta)
        # all_train_losses, all_train_accuracies,all_test_losses,all_test_accuracies,elapsed_time =None,None,None,None,None
        return epsilon,noise_multiplier, all_train_losses, all_train_accuracies,all_test_losses,all_test_accuracies,elapsed_time



class Training:
    @staticmethod
    def train(
        optimizer_type,
        model_type,
        model,
        train_loader,
        test_loader, 
        learning_rate, 
        sample_rate, 
        criterion, 
        num_epochs, 
        target_epsilon, 
        clip_bound, 
        delta, 
        device,
        accountant='prv', 
        normalize_clipping=False, 
        random_seed=474237, 
        verbose=False, **kwargs):
        """
        Main training method that integrates optimizers, privacy engines, and training loops.
        """
        # this is a known warning that can be safely ignored.
        warnings.filterwarnings(
            "ignore", message="Using a non-full backward hook when the forward contains multiple autograd Nodes"
        )
        # we would use a secure RNG in production but here we can ignore it.
        warnings.filterwarnings("ignore", message="Secure RNG turned off.")

        #check if we have the optimizer
        optimizer_type = optimizer_type.upper()
        if optimizer_type not in OptimizerRegistry:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        trainer = OptimizerRegistry[optimizer_type]()

        return trainer.train(
            model, 
            model_type,
            train_loader,  
            test_loader,
            learning_rate, 
            sample_rate, 
            criterion, 
            num_epochs, 
            target_epsilon, 
            clip_bound, 
            delta, 
            device,
            accountant=accountant,
            normalize_clipping=normalize_clipping, 
            random_seed=random_seed,
            verbose=verbose, 
            **kwargs
        )

        
