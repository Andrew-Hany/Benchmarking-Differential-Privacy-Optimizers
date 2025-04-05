from main_ import main_train_wrapper
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
#--------------------------------------------------------------------------------------------------
# Running the main function


# # Define the types for each hyperparameter
# types = [str, float, int, int, int, int, float, float, int]

# # Read the Hyperparameter combinations from the file
# with open('combinations.txt', 'r') as f:
#     combinations = [tuple(type_(value) for type_, value in zip(types, line.strip().split(','))) for line in f]

# index = int(os.environ['SLURM_ARRAY_TASK_ID'])
# if index >= len(combinations):
#     raise ValueError("SLURM_ARRAY_TASK_ID is out of range")

# optimizer, learning_rate, batch_size, epsilon, seed, num_epoch, delta, clip_bound, problem = combinations[index]


# print(f"Running with hyperparameters: optimizer={optimizer}, learning_rate={learning_rate}, batch_size={batch_size}, epsilon={epsilon}, seed={seed}, num_epochs={num_epoch}, delta={delta}, clip_bound={clip_bound}, problem={problem}")
# main_train_wrapper(
#         results_directory='results',
#         delta=delta,
#         learning_rate=learning_rate,
#         clip_bound=clip_bound,
#         batch_size=batch_size,
#         num_epochs=num_epoch,
#         target_epsilon=epsilon,
#         problem_type=problem,
#         optimizer_type=optimizer,
#         seed=seed,
# )


# This is on single GPU (not slurm)
main_train_wrapper(
    results_directory='results',
    delta=1e-5,
    learning_rate=0.01,
    clip_bound=1,
    batch_size=512,
    num_epochs=2,
    target_epsilon=5,
    problem_type=2,
    optimizer_type='sgd',
    seed=1,
)


