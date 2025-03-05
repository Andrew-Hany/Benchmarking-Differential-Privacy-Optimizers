from model_class import *
from data_class  import *

import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from transformers import ViTFeatureExtractor

def vae_loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss (BCE or MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_div

class Problem:
    def __init__(self, problem_type,batch_size):
        self.problem_type = problem_type
        self.batch_size = batch_size
        
        self.model_type = 'classification'
        self.criterion = None
        self.data_model = self.get_data_model()

    def get_data_model(self):
        if self.problem_type == 0:  # testing on Cifar 10 with simple CNN
            self.model_type = 'classification'
            self.criterion = nn.CrossEntropyLoss() 
            
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
            self.model_type = 'classification'
            self.criterion = nn.CrossEntropyLoss() 

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
            self.model_type = 'classification'
            self.criterion = nn.CrossEntropyLoss() 

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            train_loader, test_loader, classes = Data().fashion_mnist(transform, self.batch_size)

            sample_batch, _ = next(iter(train_loader))
            C, H, W = sample_batch.size(1), sample_batch.size(2), sample_batch.size(3)
            model = SimpleCNN2c2d(num_classes=len(classes), input_dims=(H, W, C))

            return train_loader, test_loader, classes, model

        elif self.problem_type ==3: # Data: Fashion-MNIST, VAE
            self.model_type = 'vae'
            self.criterion = vae_loss_function

            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            train_loader, test_loader, classes = Data().fashion_mnist(transform, self.batch_size)

            sample_batch, _ = next(iter(train_loader))
            C, H, W = sample_batch.size(1), sample_batch.size(2), sample_batch.size(3)
            model = ConvVAE(input_dims=(H, W, C), latent_dim=2) 

            return train_loader, test_loader, classes, model
        elif self.problem_type == 4: # Data: MNIST, VAE
            self.model_type = 'vae'
            self.criterion = vae_loss_function

            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            train_loader, test_loader, classes = Data().mnist(transform, self.batch_size)

            sample_batch, _ = next(iter(train_loader))
            C, H, W = sample_batch.size(1), sample_batch.size(2), sample_batch.size(3)
            model = ConvVAE(input_dims=(H, W, C), latent_dim=2) 

            return train_loader, test_loader, classes, model
        else:
            raise ValueError("Unsupported problem type")
    


        