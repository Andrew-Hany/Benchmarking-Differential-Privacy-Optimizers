import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
import numpy as n
import numpy.random as npr
import opacus
from opacus import PrivacyEngine

import warnings
from tqdm import tqdm

from trainers_class import *
from test_class import *
from problems_class import *
from Optimizers.Adam_optimizer.AdamBC import *


import torch
import torchvision
import matplotlib.pyplot as plt

def generate_and_save_images(model, data_loader, filename='generated_images.png', nrow=8):
    model.eval()
    with torch.no_grad():
        # Get a batch of images from the data loader
        data_iter = iter(data_loader)
        real_images, _ = next(data_iter)
        real_images = real_images.to(device)

        # Generate images using the model
        recon_images, _, _ = model(real_images)

    # Move images to CPU and normalize to [0, 1]
    real_images = real_images.cpu()
    recon_images = recon_images.cpu()
    real_images = (real_images - real_images.min()) / (real_images.max() - real_images.min())
    recon_images = (recon_images - recon_images.min()) / (recon_images.max() - recon_images.min())

    # Select the first nrow images from the batch
    real_image = real_images[:nrow]
    recon_image = recon_images[:nrow]

    # Concatenate real and generated images side by side
    comparison_image = torch.cat([real_image, recon_image], dim=3)  # Concatenate along width

    # Prepare the image for visualization
    comparison_image = comparison_image.permute(0, 2, 3, 1).reshape(-1, 28 * 2, 1)

    # Save the comparison image
    plt.figure(figsize=(10, 5))
    plt.imshow(comparison_image.squeeze(), cmap='gray')
    plt.axis('off')
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":

    delta = 1e-5
    learning_rate = 0.5 # Learning rate for training 3e-4
    clip_bound = 1 # Clipping norm
    batch_size = 128 # Batch size as a fraction of full data size 
    num_epochs = 4# Number of epochs

    target_epsilon = 10

    torch.manual_seed(472368)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)


    # Load data and model
    problem_module = Problem(3,batch_size)
    train_loader, test_loader,classes, model = problem_module.data_model
    model_type = problem_module.model_type # classification, VAE, ....
    criterion = problem_module.criterion  # each model should have its own criterion (loss function): 
                                        # For similicity, it will be part of the problem function
                                        # but can be modified in the main loop if needed 


    # Sample rate is needed in AdamBC
    sample_rate = batch_size / len(train_loader.dataset)

    # train model
    train_module = Training()
    epsilon,all_losses,all_accuracies = Training.train('sgd',model,train_loader,learning_rate,sample_rate,criterion,num_epochs,target_epsilon,clip_bound,
        delta, device,
        verbose=True,
        error_max_grad_norm=1.0,
        model_type = model_type
    )
    print("Epsilon: {}, Delta: {}".format(epsilon, delta))

    # # final test of test set
    test_module = Testing()
    average_loss, total_accuracy = Testing.test(model_type,model,criterion, test_loader,device,Testing.prediction_function)
    print("Loss: {}, Accuracy: {}".format(average_loss, total_accuracy))


    generate_and_save_images(model, train_loader)
    # Save the model
    # torch.save(model.state_dict(), 'trained_model.pth')
    # print("Model saved as 'trained_model.pth'")


