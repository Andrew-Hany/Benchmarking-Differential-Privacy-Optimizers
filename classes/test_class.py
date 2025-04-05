import torch
import numpy as np
from tqdm import tqdm

# Global registry for testers
_tester_registry = {}

def register_tester(model_type: str):
    """
    Decorator to register a tester class for a specific model type.
    
    Args:
        model_type (str): The type of model (e.g., "classification", "vae").
    """
    def decorator(cls):
        if model_type.lower() in _tester_registry:
            raise ValueError(f"Tester for model type '{model_type}' is already registered.")
        _tester_registry[model_type.lower()] = cls()
        return cls
    return decorator

from abc import ABC, abstractmethod
import torch
from tqdm import tqdm

class Tester(ABC):
    @abstractmethod
    def test(self, model, criterion, test_loader, device):
        pass

@register_tester("classification")
class ClassificationTester(Tester):
    def __init__(self):
        self.prediction_function = lambda output: torch.argmax(output, dim=1)

    def test(self, model, criterion, test_loader, device):
        model.eval()
        losses, accuracies, total_size = [], [], 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                total_size += x.size(0)
                out = model(x)
                loss = criterion(out, y)
                preds = self.prediction_function(out)
                corrects = torch.sum(preds == y).item()
                losses.append(loss.item() * x.size(0))
                accuracies.append(corrects)
        avg_loss = sum(losses) / total_size
        accuracy = sum(accuracies) / total_size
        return avg_loss, accuracy

@register_tester("vae")
class VAETester(Tester):
    def test(self, model, criterion, test_loader, device):
        model.eval()
        losses, total_size = [], 0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                total_size += len(x)
                recon_x, mu, logvar = model(x)
                loss = criterion(recon_x, x, mu, logvar)
                losses.append(loss.item() * x.size(0))
        avg_loss = sum(losses) / total_size
        return avg_loss, None
 
class TestManager:
    def test(self, model_type: str, model, criterion, test_loader, device):
        """
        Perform testing using the appropriate tester based on the model type.
        
        Args:
            model_type (str): The type of model (e.g., "classification", "vae").
            model: The PyTorch model to test.
            criterion: The loss function.
            test_loader: The data loader for the test set.
            device: The device (CPU or GPU) to run the test on.
        
        Returns:
            The results of the test (e.g., average loss, accuracy).
        """
        tester = _tester_registry.get(model_type.lower())
        if not tester:
            raise ValueError(f"No tester registered for model type: {model_type}")
        return tester.test(model, criterion, test_loader, device)