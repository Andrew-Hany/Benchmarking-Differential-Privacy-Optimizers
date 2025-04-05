from abc import ABC, abstractmethod
import torchvision.datasets as datasets
import torch.utils.data
import torchvision.transforms as transforms

# Abstraction for Data Providers
class DataProvider(ABC):
    @abstractmethod
    def get_data(self, transform=None) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, list]:
        """
        Returns train_loader, test_loader, and classes.
        If no transform is provided, uses the default transformation for the dataset.
        """
        pass

# Concrete Implementation for CIFAR-10
class CIFAR10Provider(DataProvider):
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def get_default_transform(self):
        """Default transformation for CIFAR-10."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def get_data(self, transform=None) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, list]:
        transform = transform if transform else self.get_default_transform()

        # Load training data
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)

        # Load testing data
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=2)

        # Define classes
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return trainloader, testloader, classes

# Concrete Implementation for CIFAR-100
class CIFAR100Provider(DataProvider):
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def get_default_transform(self):
        """Default transformation for CIFAR-100."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

    def get_data(self, transform=None) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, list]:
        transform = transform if transform else self.get_default_transform()

        # Load training data
        trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)

        # Load testing data
        testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=2)

        # Define classes
        classes = [f'class_{i}' for i in range(100)]  # CIFAR-100 has 100 classes
        return trainloader, testloader, classes

# Concrete Implementation for MNIST
class MNISTProvider(DataProvider):
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def get_default_transform(self):
        """Default transformation for MNIST."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def get_data(self, transform=None) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, list]:
        transform = transform if transform else self.get_default_transform()

        # Load training data
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)

        # Load testing data
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=2)

        # Define classes
        classes = [str(i) for i in range(10)]  # MNIST has 10 classes (digits 0-9)
        return trainloader, testloader, classes

# Concrete Implementation for FashionMNIST
class FashionMNISTProvider(DataProvider):
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def get_default_transform(self):
        """Default transformation for FashionMNIST."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def get_data(self, transform=None) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, list]:
        transform = transform if transform else self.get_default_transform()

        # Load training data
        trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)

        # Load testing data
        testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=2)



        # Define classes
        classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        return trainloader, testloader, classes