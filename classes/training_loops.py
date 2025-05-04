from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import warnings
import opacus
import sys
import os
import time

from opacus.utils.batch_memory_manager import BatchMemoryManager

from .test_class import *
# Registry for training loops
TrainingLoopRegistry = {}

def register_training_loop(model_type):
    def decorator(cls):
        TrainingLoopRegistry[model_type] = cls
        return cls
    return decorator


# Abstract base class for training loops
class BaseTrainingLoop(ABC):
    @abstractmethod
    def train(self, num_epochs, train_loader, test_loader, model, criterion, optimizer, device, verbose=False, use_closure=False):
        pass



@register_training_loop("classification")
class ClassificationTrainingLoop(BaseTrainingLoop):
    def train(self, num_epochs, train_loader,test_loader, model, criterion, optimizer, device, verbose=False, use_closure=False):
        all_train_losses = []       # Per epoch losses (list of lists)
        all_train_accuracies = []
        all_test_losses = []
        all_test_accuracies = []
        
        # Initialize TestManager
        test_module = TestManager()

        model.to(device)
        for epoch in range(num_epochs):
            model.train()

            logical_batch_losses = []
            logical_batch_accuracies = []
            physical_batch_losses = []
            physical_batch_accuracies = []  
            with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=256,
                optimizer=optimizer
            ) as memory_safe_loader:
                for x, y in tqdm(memory_safe_loader, desc=f'{epoch+1}/{num_epochs}'):
                    x, y = x.to(device), y.to(device)

                    if use_closure:
                        def closure():
                            out = model(x)  # Forward pass
                            loss = criterion(out, y)  # Compute the loss
                            loss.backward()  # Backward pass
                            return loss, out
                        loss, out = optimizer.step(closure)
                        optimizer.zero_grad()

                    else:
                        out = model(x)
                        loss = criterion(out, y)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    # Record loss and accuracy
                    physical_batch_losses.append(loss.item())
                    _, predicted = torch.max(out.data, 1)
                    accuracy = (predicted == y).sum().item() / y.size(0)
                    physical_batch_accuracies.append(accuracy)
                    
                    # If this was the last step in a logical batch
                    # _is_last_step_skipped checks if .step is called or not
                    # if it is not skipped (false), it is the logical batch
                    if not getattr(optimizer, "_is_last_step_skipped", False): 

                        avg_loss = sum(physical_batch_losses) / len(physical_batch_losses)
                        avg_acc = sum(physical_batch_accuracies) / len(physical_batch_accuracies)
                        logical_batch_losses.append(avg_loss)
                        logical_batch_accuracies.append(avg_acc)

                        # Clear physical batch trackers
                        physical_batch_losses = []
                        physical_batch_accuracies = []

                # Save training metrics
                all_train_losses.append(logical_batch_losses)
                all_train_accuracies.append(logical_batch_accuracies)

                # Perform testing after each epoch

                test_loss, test_accuracy = test_module.test("classification", model, criterion, test_loader, device)
                all_test_losses.append(test_loss)
                all_test_accuracies.append(test_accuracy)
                if verbose:
                    print(f"Epoch {epoch + 1}, loss  = {np.sum(logical_batch_losses) / len(logical_batch_losses) }, accuracy = {np.sum(logical_batch_accuracies) / len(logical_batch_accuracies)},Test Loss = {test_loss}, Test Accuracy = {test_accuracy}")

        return all_train_losses, all_train_accuracies,all_test_losses,all_test_accuracies

@register_training_loop("vae")
class VAETrainingLoop(BaseTrainingLoop):
    def train(self, num_epochs, train_loader,test_loader, model, criterion, optimizer, device, verbose=False, use_closure=False):
        all_train_losses = []
        all_test_losses = []
        
        # Initialize TestManager
        test_module = TestManager()
        model.to(device)
        for epoch in range(num_epochs):
            model.train()
            
            logical_batch_losses = []
            physical_batch_losses = []

            with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=256,
                optimizer=optimizer
            ) as memory_safe_loader:
                for x, _ in tqdm(memory_safe_loader, desc=f'{epoch+1}/{num_epochs}'):
                    x = x.to(device)
                    if use_closure:
                        def closure():
                            recon_x, mu, logvar = model(x)  # Forward pass
                            loss = criterion(recon_x, x, mu, logvar)  # Compute the loss
                            loss.backward()  # Backward pass
                            return loss, recon_x
                        loss, recon_x = optimizer.step(closure)
                        optimizer.zero_grad()
                    else:
                        recon_x, mu, logvar = model(x)
                        loss = criterion(recon_x, x, mu, logvar)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    physical_batch_losses.append(loss.item())

                    # If this was the last step in a logical batch
                    # _is_last_step_skipped checks if .step is called or not
                    # if it is not skipped (false), it is the logical batch
                    if not getattr(optimizer, "_is_last_step_skipped", False): 
                        avg_loss = sum(physical_batch_losses) / len(physical_batch_losses)
                        logical_batch_losses.append(avg_loss)
                        

                # Save training metrics
                all_train_losses.append(logical_batch_losses)
                # Perform testing after each epoch
                test_loss, _ = test_module.test("vae", model, criterion, test_loader, device)
                all_test_losses.append(test_loss)

                if verbose:
                    print(f"Epoch {epoch + 1}, Loss = {np.mean(logical_batch_losses)},Test Loss = {test_loss}")
        return all_train_losses, [],all_test_losses,[]

# Training Orchestrator
class TrainingOrchestrator:
    def training_loop(self, model_type, num_epochs, train_loader,test_loader, model, criterion, optimizer, device, verbose=False, use_closure=False):
        """
        Main training loop that delegates to specific training loops based on model type.
        """
        # Resolve the training loop from the registry
        if model_type.lower() not in TrainingLoopRegistry:
            raise ValueError(f"Unsupported model type: {model_type}")
        training_loop_class = TrainingLoopRegistry[model_type.lower()]
        training_loop = training_loop_class()

        # Start training
        start_time = time.time()
        all_train_losses, all_train_accuracies,all_test_losses,all_test_accuracies = training_loop.train(
            num_epochs=num_epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            verbose=verbose,
            use_closure=use_closure
        )
        elapsed_time = time.time() - start_time
        return all_train_losses, all_train_accuracies,all_test_losses,all_test_accuracies, elapsed_time