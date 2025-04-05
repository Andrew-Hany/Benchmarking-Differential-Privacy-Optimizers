import os

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
from opacus.accountants.utils import get_noise_multiplier

import warnings
from tqdm import tqdm

from classes.trainers_class import *
from classes.test_class import *
from classes.problems_class import *
from Optimizers.Adam_optimizer.AdamBC import *
from classes.saving_class import *
from classes.reporting_class import *
from classes.keep_track_class import *
from transformers import *


import torch
import torchvision
import matplotlib.pyplot as plt

from itertools import product
from queue import Queue
from threading import Thread

import time
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


def main_train_wrapper(
    results_directory,
    delta = 1e-5,
    learning_rate = 0.1 ,# Learning rate for training 3e-4
    clip_bound = 1 ,# Clipping norm
    batch_size = 128 ,# Batch size as a fraction of full data size 
    num_epochs = 1,# Number of epochs
    target_epsilon = 10,
    problem_type=0,
    optimizer_type ='sgd',
    seed =474237,
    error_max_grad_norm = 1 # for Dice Optimizer

):

    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    problem_module = Problem(problem_type, batch_size)
    train_loader = problem_module.train_loader
    test_loader = problem_module.test_loader
    model = problem_module.model
    model_type = problem_module.model_type # classification, VAE, ....
    criterion = problem_module.criterion  # each model should have its own criterion (loss function): 
                                        # For similicity, it will be part of the problem function
                                        # but can be modified in the main loop if needed 

    # Sample rate is needed in AdamBC
    sample_rate = batch_size / len(train_loader.dataset)

    # train model
    train_module = Training()
    epsilon,noise_multiplier,all_losses,all_accuracies,elapsed_time = Training.train(optimizer_type,model_type,model,train_loader,learning_rate,sample_rate,criterion,num_epochs,target_epsilon,clip_bound,
        delta, device,
        normalize_clipping= True,
        random_seed = seed,
        verbose=True,
        error_max_grad_norm=error_max_grad_norm
        
    )

    print("Epsilon: {}, Delta: {}, Time Taken for Training {}".format(epsilon, delta,elapsed_time))

    #final test of test set
    test_module = Testing()
    average_loss, total_accuracy = Testing.test(model_type,model,criterion, test_loader,device,Testing.prediction_function)
    print("Loss: {}, Accuracy: {}".format(average_loss, total_accuracy))

    if model_type.lower() == 'vae':
        generate_and_save_images(model, train_loader)

    # add noise multiplier
    parameters = {
        "optimizer_type": optimizer_type,
        "model_type" : model_type,
        "problem_type": problem_type,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,  # Number of epochs 
        "batch_size": batch_size,  # Batch size as a fraction of full data size
        "sample_rate":sample_rate,
        "epsilon": target_epsilon,
        "noise_multiplier": noise_multiplier,
        "delta": delta,
        "clip_bound": clip_bound,  # Clipping norm
        "seed": seed,
    }

    # Conditionally add "error_max_grad_norm" if optimizer_type is 'dice'
    if optimizer_type.lower() == 'dice':
        parameters["error_max_grad_norm"] = 1  # for Dice Optimizer

    #saving the files and the results into .pth and .json
    saving_module = Saving()
    results = saving_module.save_results(results_directory,model,parameters,all_losses,all_accuracies,average_loss, total_accuracy,elapsed_time)


def reporting_wrapper(results_directory):
    # Saving all json files into one CSV file:
    saving_module = Saving()
    csv_file_path = saving_module.convert_json_to_csv(results_directory)

    hyperparameters = ["optimizer_type","problem_type","learning_rate","num_epochs","batch_size","epsilon","delta","clip_bound","seed"]
    tracker = HyperparameterTracker(csv_file_path,hyperparameters,'/scratch/project_2003275/Andrew_temp/Benchmarking-Differential-Privacy-Optimizers/results/hyperparameters_tracking.csv')
    tracker_file_name = tracker.extract_hyperparameters()

    # here I should Add the visualization reports functions: These functions should call from reporting class
    pass
#--------------------------------------------------------------------------------------------------
# Running the main function

# hyperparameters = ["optimizer_type", "problem_type", "learning_rate", "num_epochs", "batch_size", "epsilon", "delta", "clip_bound", "seed"]
# tracker = HyperparameterTracker('/scratch/project_2003275/Andrew_temp/Benchmarking-Differential-Privacy-Optimizers/hyperparameter_combinations_generate.py',hyperparameters,'/scratch/project_2003275/Andrew_temp/Benchmarking-Differential-Privacy-Optimizers/results/hyperparameters_tracking.csv')

# optimizer_types = ['sgd']
# # learning_rates = np.logspace(np.log10(0.1), np.log10(0.7), num=4) #sgd
# learning_rates = [0.001, 0.01 , 0.1  , 1.   ]
# batch_sizes = [128, 256]
# epsilons = [1, 5, 10]
# seeds = [472368]

# num_epochs = [10]
# deltas = [1e-5]
# clip_bounds = [1]
# problem_types = [1]
# torch.cuda.empty_cache()

# for optimizer in optimizer_types:
#     for problem in problem_types:
#         for lr in learning_rates:
#             for epoch in num_epochs:
#                 for batch_size in batch_sizes:
#                     for epsilon in epsilons:
#                         for delta in deltas:
#                             for clip_bound in clip_bounds:
#                                 for seed in seeds:
#                                     if not tracker.has_run(optimizer_type=optimizer, problem_type=problem, learning_rate=lr, num_epochs=epoch, batch_size=batch_size, epsilon=epsilon, delta=delta, clip_bound=clip_bound, seed=seed):
#                                         main_train_wrapper(
#                                             results_directory='results',
#                                             delta=delta,
#                                             learning_rate=lr,
#                                             clip_bound=clip_bound,
#                                             batch_size=batch_size,
#                                             num_epochs=epoch,
#                                             target_epsilon=epsilon,
#                                             problem_type=problem,
#                                             optimizer_type=optimizer,
#                                             seed=seed,
#                                         )
             
#                                     else:
#                                         print('done before')
#                                         print(optimizer, epsilon, batch_size, lr, seed, epoch, problem)


# reporting_wrapper("results")

# main_train_wrapper(
#     results_directory='results',
#     delta=1e-5,
#     learning_rate=0.01,
#     clip_bound=1,
#     batch_size=1024,
#     num_epochs=1,
#     target_epsilon=5,
#     problem_type=1,
#     optimizer_type='sgd',
#     seed=1,
# )
