from model_class import *
from data_class  import *

import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from transformers import ViTFeatureExtractor
class Problem:
    def __init__(self, problem_type,batch_size):
        self.problem_type = problem_type
        self.batch_size = batch_size
        self.data_model = self.get_data_model()

    def get_data_model(self):
        if self.problem_type == 0:  # testing on Cifar 10 with simple CNN
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            train_loader, test_loader,classes = Data().cifar10(transform,self.batch_size)
            sample_batch, _ = next(iter(train_loader))
            
            C, H, W = sample_batch.size(1), sample_batch.size(2), sample_batch.size(3)
            model =  CNNNet(len(classes), input_dims=( H, W,C))
            
            return  train_loader, test_loader,classes,  model
        elif self.problem_type == 1: # Data:cifar 10 , Simple CNN: 3c3d
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            train_loader, test_loader,classes = Data().cifar10(transform,self.batch_size)

            sample_batch, _ = next(iter(train_loader))
            C, H, W = sample_batch.size(1), sample_batch.size(2), sample_batch.size(3)
            model =  SimpleCNN3c3d(len(classes),  input_dims=( H, W,C))

            return  train_loader, test_loader,classes,  model
        elif self.problem_type == 2:  # Data: Fashion-MNIST, Simple CNN: 2c2d
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            train_loader, test_loader, classes = Data().fashion_mnist(transform, self.batch_size)

            sample_batch, _ = next(iter(train_loader))
            C, H, W = sample_batch.size(1), sample_batch.size(2), sample_batch.size(3)
            model = SimpleCNN2c2d(num_classes=len(classes), input_dims=(H, W, C))

            return train_loader, test_loader, classes, model
        else:
            raise ValueError("Unsupported problem type")


        