from typing import IO, Any, BinaryIO, Dict, List, Optional, Tuple, Union
import os
import warnings
from itertools import chain

from opacus.optimizers import DPOptimizer
from opacus.privacy_engine import PrivacyEngine

import os
import warnings
from itertools import chain
from typing import IO, Any, BinaryIO, Dict, List, Optional, Tuple, Union

import torch
from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.data_loader import DPDataLoader, switch_generator
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.grad_sample import (
    AbstractGradSampleModule,
    GradSampleModule,
    get_gsm_class,
    wrap_model,
)

from opacus.schedulers import _GradClipScheduler, _NoiseScheduler
from opacus.validators.module_validator import ModuleValidator
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from Optimizers.Matrix_optimizer._init import *
from Optimizers.Matrix_optimizer.MatrixSGD import DPOptimizer_Matrix

import numpy as np
class Matrix_single_epoch_lambda_PrivacyEngine(PrivacyEngine):
    def __init__(self, *, accountant: str = "prv", secure_mode: bool = False):
        super().__init__(accountant=accountant, secure_mode=secure_mode)

    def make_private_with_epsilon(
        self,
        *,
        module: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: DataLoader,
        target_epsilon: float,
        target_delta: float,
        epochs: int,
        max_grad_norm: Union[float, List[float]],
        batch_first: bool = True,
        loss_reduction: str = "mean",
        poisson_sampling: bool = False,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode: str = "hooks",
        normalize_clipping: bool = False,
        total_steps: int = None,
        **kwargs,
    ):
        # for matrix, we need to use gdp
        noise_multiplier = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=1,
            epochs=1,
            accountant=self.accountant.mechanism(),
            **kwargs,
        )

        return self.make_private(
            module=module,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            noise_generator=noise_generator,
            grad_sample_mode=grad_sample_mode,
            poisson_sampling=poisson_sampling,
            clipping=clipping,
            normalize_clipping=normalize_clipping,
            total_steps=total_steps,
            epochs=epochs
        )

    def make_private(
        self,
        *,
        module: nn.Module,
        optimizer: optim.Optimizer,
        criterion=nn.CrossEntropyLoss(),  # Added deafult for backward compatibility
        data_loader: DataLoader,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        batch_first: bool = True,
        loss_reduction: str = "mean",
        poisson_sampling: bool = False,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode: str = "hooks",
        **kwargs,
    ) -> Tuple[GradSampleModule, DPOptimizer, DataLoader]:
    
        if noise_generator and self.secure_mode:
            raise ValueError("Passing seed is prohibited in secure mode")

        # compare module parameter with optimizer parameters
        model_parameters = set(module.parameters())
        for p in chain.from_iterable(
            [param_group["params"] for param_group in optimizer.param_groups]
        ):
            if p not in model_parameters:
                raise ValueError(
                    "Module parameters are different than optimizer Parameters"
                )

        distributed = isinstance(module, (DPDDP, DDP))

        module = self._prepare_model(
            module,
            batch_first=batch_first,
            max_grad_norm=max_grad_norm,
            loss_reduction=loss_reduction,
            grad_sample_mode=grad_sample_mode,
        )
        if poisson_sampling:
            module.forbid_grad_accumulation()

        data_loader = self._prepare_data_loader(
            data_loader, distributed=distributed, poisson_sampling=poisson_sampling
        )

        optimizer_steps_per_epoch = len(data_loader) # This is new to get the steps per epoch
        sample_rate = 1 / len(data_loader)
        expected_batch_size = int(len(data_loader.dataset) * sample_rate)

        # expected_batch_size is the *per worker* batch size
        if distributed:
            world_size = torch.distributed.get_world_size()
            expected_batch_size /= world_size

        kwargs['optimizer_steps_per_epoch']=optimizer_steps_per_epoch
        optimizer = self._prepare_optimizer(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            noise_generator=noise_generator,
            distributed=distributed,
            clipping=clipping,
            grad_sample_mode=grad_sample_mode,
            
            **kwargs,
        )

        # optimizer.attach_step_hook(
        #     self.accountant.get_optimizer_hook_fn(sample_rate=sample_rate)
        # )

        # We log the noise once for matrix multiplication
        self.accountant.history.append((noise_multiplier, 1, 1))
        
        if grad_sample_mode == "ghost":
            criterion = self._prepare_criterion(
                module=module,
                optimizer=optimizer,
                criterion=criterion,
                loss_reduction=loss_reduction,
                **kwargs,
            )

            return module, optimizer, criterion, data_loader

        return module, optimizer, data_loader

    def _prepare_optimizer(
        self,
        *,
        optimizer: optim.Optimizer,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        expected_batch_size: int,
        loss_reduction: str = "mean",
        distributed: bool = False,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode="hooks",
        normalize_clipping: bool = False,       
        **kwargs,
    ) -> DPOptimizer:
        if isinstance(optimizer, DPOptimizer_Matrix):
            optimizer = optimizer.original_optimizer


        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_generator is not None:
            generator = noise_generator

        optim_class = get_optimizer_class()


        
        # A = low triangle (TxT)
        # B = A (TxT)
        # C = I (TxT)

        T_calculated =kwargs['epochs']*kwargs['optimizer_steps_per_epoch']
        print(kwargs['epochs'],kwargs['optimizer_steps_per_epoch'])
        
        # C_matrix_T_by_T = torch.eye(T_calculated, device='cuda').to('cpu') if torch.cuda.is_available() else torch.eye(T_calculated, device='cpu')

        B_path = f'B_{T_calculated}_lambda_matrix.npy'
        if os.path.exists(B_path):
            print(f"[SKIP] File already exists: {B_path}")
        else:
            B_matrix_T_by_T, _ = get_matrix_B_and_C_single_epoch_fixed_point(T_calculated, lamda_matrix=True)
            sens_C = compute_sensitivity(None) # This will be one

            # B_matrix_T_by_T = torch.tril(torch.ones(T_calculated, T_calculated, device='cpu'))
            B_np = B_matrix_T_by_T.cpu().numpy()
            np.save(B_path, B_np)
            
            del B_matrix_T_by_T
        return optim_class(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=self.secure_mode,
            normalize_clipping=normalize_clipping,
            B_matrix =None,
            B_path = B_path,
            sens_C=1,
            lamda_matrix=True,
            **kwargs,
        )
