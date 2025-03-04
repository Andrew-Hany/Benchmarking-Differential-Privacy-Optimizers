from model_class import *
from data_class  import *

import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from transformers import ViTFeatureExtractor
class Problem:
    def __init__(self, problem_type,sample_rate):
        self.problem_type = problem_type
        self.sample_rate = sample_rate
        self.data_model = self.get_data_model()

    def get_data_model(self):
        if self.problem_type == 0:  # testing on Cifar 10 with simple CNN
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            train_loader, test_loader,classes = Data().cifar10(transform,self.sample_rate)
            return  train_loader, test_loader,classes, CNNNet(len(classes))
        elif self.problem_type == 1: # Data:cifar 10 , Simple CNN: 3c3d
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            train_loader, test_loader,classes = Data().cifar10(transform,self.sample_rate)
            return  train_loader, test_loader,classes,  SimpleCNN3c3d(len(classes))
        
        else:
            raise ValueError("Unsupported problem type")


        