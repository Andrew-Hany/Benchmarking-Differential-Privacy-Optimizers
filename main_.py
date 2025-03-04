import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
import cv2
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




if __name__ == "__main__":
    delta = 1e-5
    learning_rate = 0.05 # Learning rate for training
    clip_bound = 1 # Clipping norm
    sample_rate = 0.03 # Batch size as a fraction of full data size 
    num_epochs = 1 # Number of epochs
    torch.manual_seed(472368)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    target_epsilon = 10

    # Load data and model
    problem_module = Problem(0,sample_rate)
    train_loader, test_loader,classes, model = problem_module.data_model

    criterion = nn.CrossEntropyLoss()

    # train model
    train_module = Training()
    epsilon,all_losses,all_accuracies = Training.train('sgd',model,train_loader,learning_rate,sample_rate,criterion,num_epochs,target_epsilon,clip_bound,
        delta, device,
        verbose=True,
        error_max_grad_norm=1.0,
    )
    print("Epsilon: {}, Delta: {}".format(epsilon, delta))

    # # final test of test set
    test_module = Testing()
    average_loss, total_accuracy = Testing.test(model,criterion, test_loader,device,Testing.prediction_function)
    print("Loss: {}, Accuracy: {}".format(average_loss, total_accuracy))

    # Save the model
    # torch.save(model.state_dict(), 'trained_model.pth')
    # print("Model saved as 'trained_model.pth'")


