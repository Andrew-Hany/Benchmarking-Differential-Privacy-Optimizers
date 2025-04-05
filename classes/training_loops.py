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
    def train(self, num_epochs, train_loader, model, criterion, optimizer, device, verbose=False, use_closure=False):
        pass

@register_training_loop("classification")
class ClassificationTrainingLoop(BaseTrainingLoop):
    def train(self, num_epochs, train_loader, model, criterion, optimizer, device, verbose=False, use_closure=False):
        all_losses = []
        all_accuracies = []
        model.to(device)
        for epoch in range(num_epochs):
            epoch_losses = []
            epoch_accuracies = []
            for x, y in tqdm(train_loader, desc=f'{epoch+1}/{num_epochs}'):
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

                epoch_losses.append(loss.item())
                # Calculate accuracy
                _, predicted = torch.max(out.data, 1)
                accuracy = (predicted == y).sum().item() / y.size(0)
                epoch_accuracies.append(accuracy)

            all_losses.append(epoch_losses)
            all_accuracies.append(epoch_accuracies)
            if verbose:
                print(f"Epoch {epoch + 1}, loss  = {np.sum(epoch_losses) / len(epoch_losses) }, accuracy = {np.sum(epoch_accuracies) / len(epoch_accuracies)}")

        return all_losses, all_accuracies

@register_training_loop("vae")
class VAETrainingLoop(BaseTrainingLoop):
    def train(self, num_epochs, train_loader, model, criterion, optimizer, device, verbose=False, use_closure=False):
        all_losses = []
        model.to(device)
        for epoch in range(num_epochs):
            epoch_losses = []
            for x, _ in tqdm(train_loader, desc=f'{epoch+1}/{num_epochs}'):
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
                epoch_losses.append(loss.item())
            all_losses.append(epoch_losses)
            if verbose:
                print(f"Epoch {epoch + 1}, Loss = {np.mean(epoch_losses)}")
        return all_losses, []

# Training Orchestrator
class TrainingOrchestrator:
    def training_loop(self, model_type, num_epochs, train_loader, model, criterion, optimizer, device, verbose=False, use_closure=False):
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
        all_losses, all_accuracies = training_loop.train(
            num_epochs=num_epochs,
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            verbose=verbose,
            use_closure=use_closure
        )
        elapsed_time = time.time() - start_time
        return all_losses, all_accuracies, elapsed_time