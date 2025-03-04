

####################
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from transformers import ViTFeatureExtractor

class Data:
    # @staticmethod
    # def get_transform(model_type):
    #     if model_type == 'vit':
    #         feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    #         return transforms.Compose([
    #             transforms.Resize((224, 224)),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    #         ])
    #     elif model_type == 'cnn':
    #         return transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #         ])
    #     elif model_type == 'mnist':
    #         return transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.1307,), (0.3081,))
    #         ])
    #     else:
    #         raise ValueError("Unsupported model type")

    @staticmethod
    def cifar10(transform,sample_rate):
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        train_batch_size = int(len(trainset) * sample_rate)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)

        
        

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_batch_size = int(len(testset) * sample_rate)
        testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        return trainloader, testloader, classes

    @staticmethod
    def cifar100(transform,sample_rate):
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        train_batch_size = int(len(trainset) * sample_rate)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        test_batch_size = int(len(testset) * sample_rate)
        testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)

        classes = [f'class_{i}' for i in range(100)]  # CIFAR-100 has 100 classes

        return trainloader, testloader, classes

    @staticmethod
    def mnist(transform,sample_rate):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_batch_size = int(len(trainset) * sample_rate)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_batch_size = int(len(testset) * sample_rate)
        testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)

        classes = [str(i) for i in range(10)]  # MNIST has 10 classes (digits 0-9)

        return trainloader, testloader, classes