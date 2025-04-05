from .data_class import *
from .model_class import *
from .transforms import TransformFactory
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

# Registry for Problem Definitions
_problem_registry = {}

def register_problem(problem_type: int):
    """
    Args:
        problem_type (int): The unique identifier for the problem.
    """
    def decorator(cls):
        if problem_type in _problem_registry:
            raise ValueError(f"Problem type {problem_type} already registered.")
        _problem_registry[problem_type] = cls()
        return cls
    return decorator

# Base Class for Problem Definitions
class ProblemDefinition(ABC):
    """
    Base class for problem definitions.
    Each problem type must define:
    - data_provider: The dataset provider.
    - model_factory: The model factory.
    - kwargs: Additional arguments for the model factory.
    - model_type: The type of model (e.g., 'classification', 'vae').
    - criterion: The loss function.
    """
    @property
    @abstractmethod
    def data_provider(self):
        pass

    @property
    @abstractmethod
    def model_factory(self):
        pass

    @property
    @abstractmethod
    def kwargs(self):
        pass

    @property
    @abstractmethod
    def model_type(self):
        pass

    @property
    @abstractmethod
    def criterion(self):
        pass
    @property
    @abstractmethod
    def problem_name(self):
        """A unique name/identifier for the problem."""
        pass

# Problem 0: Testing Problem -> CIFAR-10 Classification with CNNNet
@register_problem(0)
class Problem0(ProblemDefinition):
    @property
    def data_provider(self):
        return CIFAR10Provider

    @property
    def model_factory(self):
        return CNNNetFactory

    @property
    def kwargs(self):
        return {}

    @property
    def model_type(self):
        return "classification"

    @property
    def criterion(self):
        return nn.CrossEntropyLoss()
    @property
    def problem_name(self):
        return "cifar10_cnnnet"

# Problem 1: CIFAR-10 Classification with SimpleCNN3c3d
@register_problem(1)
class Problem1(ProblemDefinition):
    @property
    def data_provider(self):
        return CIFAR10Provider

    @property
    def model_factory(self):
        return SimpleCNN3c3dFactory

    @property
    def kwargs(self):
        return {}

    @property
    def model_type(self):
        return "classification"

    @property
    def criterion(self):
        return nn.CrossEntropyLoss()
    @property
    def problem_name(self):
        return "cifar10_3c3d"

# Problem 2: Fashion-MNIST Classification with SimpleCNN2c2d
@register_problem(2)
class Problem2(ProblemDefinition):
    @property
    def data_provider(self):
        return FashionMNISTProvider

    @property
    def model_factory(self):
        return SimpleCNN2c2dFactory

    @property
    def kwargs(self):
        return {}

    @property
    def model_type(self):
        return "classification"

    @property
    def criterion(self):
        return nn.CrossEntropyLoss()
    @property
    def problem_name(self):
        return "fashion_mnist_2c2d"

# Problem 3: Fashion-MNIST VAE
@register_problem(3)
class Problem3(ProblemDefinition):
    @property
    def data_provider(self):
        return FashionMNISTProvider

    @property
    def model_factory(self):
        return ConvVAEFactory

    @property
    def kwargs(self):
        return {"latent_dim": 2}

    @property
    def model_type(self):
        return "vae"

    @property
    def criterion(self):
        def vae_loss_function(recon_x, x, mu, logvar):
            recon_loss = F.mse_loss(recon_x, x, reduction='sum')
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + kl_div
        return vae_loss_function
    @property
    def problem_name(self):
        return "fashion_mnist_vae"

# Problem 4: MNIST VAE
@register_problem(4)
class Problem4(ProblemDefinition):
    @property
    def data_provider(self):
        return MNISTProvider

    @property
    def model_factory(self):
        return ConvVAEFactory

    @property
    def kwargs(self):
        return {"latent_dim": 2}

    @property
    def model_type(self):
        return "vae"

    @property
    def criterion(self):
        def vae_loss_function(recon_x, x, mu, logvar):
            recon_loss = F.mse_loss(recon_x, x, reduction='sum')
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + kl_div
        return vae_loss_function
    @property
    def problem_name(self):
        return "mnist_vae"

#Problem Class
class Problem:
    def __init__(self, problem_type: int, batch_size: int):
        self.problem_type = problem_type
        self.batch_size = batch_size

        # Resolve the problem definition from the registry
        if problem_type not in _problem_registry:
            raise ValueError(f"No problem registered for type: {problem_type}")
        problem_definition = _problem_registry[problem_type]

        # Initialize data provider
        self.data_provider = problem_definition.data_provider(batch_size=batch_size)

        # Initialize model factory
        self.model_factory = problem_definition.model_factory(**problem_definition.kwargs)

        # Combine transformations
        dataset_transform = self.data_provider.get_default_transform()
        model_transform = self.model_factory.get_transform()
        combined_transform = TransformFactory.combine_transforms(dataset_transform, model_transform)

        # Get data
        self.train_loader, self.test_loader, self.classes = self.data_provider.get_data(transform=combined_transform)

        # Create model
        sample_batch, _ = next(iter(self.train_loader))
        C, H, W = sample_batch.size(1), sample_batch.size(2), sample_batch.size(3)
        self.model = self.model_factory.create_model(len(self.classes), (H, W, C))

        # Set model type and criterion
        self.model_type = problem_definition.model_type
        self.criterion = problem_definition.criterion