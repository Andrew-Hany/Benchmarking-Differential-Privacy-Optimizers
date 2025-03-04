# import torch
# import torchvision
# import torchvision.transforms as transforms
# import torch.nn as nn

# import torch.nn.functional as F
# import torch.optim as optim
# import cv2
# import numpy as np
# import numpy.random as npr
# import opacus
# from opacus import PrivacyEngine

# import warnings
# from tqdm import tqdm

# from trainers_class import *
# from test import *
# from data import *
# from model import *
# from Optimizers.AdamBC import *
# from Optimizers.Dice_optimizers.DiceSGD import *


# import copy
# import pandas as pd

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# sample_size = 50000
# class Benchmark():
#     def __init__(self,model_type,intialize_new_optimizer):
            
#         self.optimizers_models_dic = {}
        
#         model1 = Model(model_type,len(classes)).model
#         optimizer1 = optim.SGD(model1.parameters(), learning_rate)
    
#         model2 = Model(model_type,len(classes)).model
#         optimizer2 = AdamCorr(
#                             model2.parameters(), lr=learning_rate, eps=1e-8,
#                             dp_batch_size=int(len(train_loader.dataset) * sample_rate),
#                             dp_noise_multiplier=noise_multiplier,
#                             dp_l2_norm_clip=clip_bound,
#                             eps_root=1e-8,
#                             betas=(0.9, 0.999),
#                             gamma_decay=0.99
#                         )
        
#         self.optimizers_models_dic= \
#         {
#             'sgd':
#                 {
#                     'model':model1,
#                     'optimizer':optimizer1
#                 },
#             'adamcorr':
#                 {
#                     'model':model2,
#                     'optimizer':optimizer2
#                 }
#         }   

#         if intialize_new_optimizer:
#             model3 = Model(model_type,len(classes)).model
#             new_optimizer = intialize_new_optimizer(model3)
#             self.optimizers_models_dic['new_optimizer']=\
#                 {                                        
#                     'model':model3,
#                     'optimizer':new_optimizer
#                 }

#     def benchmark(self,train_loader, test_loader, learning_rate, sample_rate, noise_multiplier, clip_bound, delta, num_epochs):
#         '''
#         compare specific optimizer to existing ones on a specific dataset and specifc model
#         '''
#         res = {}
#         criterion = nn.CrossEntropyLoss()
#         for optimizer_name,optimizer_dic in self.optimizers_models_dic.items():
#             model = optimizer_dic['model']
#             optimizer = optimizer_dic['optimizer']

#             train_module = Training()
#             epsilon = Training.private_train(
#                 model, train_loader, optimizer,criterion, learning_rate, num_epochs, noise_multiplier, clip_bound, delta,verbose=True
#             )
#             print("Epsilon: {}, Delta: {}".format(epsilon, delta))

#             # # final test of test set
#             test_module = Testing()
#             average_loss, total_accuracy = Testing.test(model,criterion, test_loader,Testing.prediction_function)
#             print("Loss: {}, Accuracy: {}".format(average_loss, total_accuracy))

#             res[optimizer_name] = {}
#             res[optimizer_name]['Epsilon']=epsilon
#             res[optimizer_name]['average_loss']=average_loss
#             res[optimizer_name]['total_accuracy']=total_accuracy
#             res[optimizer_name]['delta']=delta

#         self.print_results(res)
#         return res

#     def print_results(self,res):
#         df = pd.DataFrame(res)
#         print(df)

# # ___________________________________________________________________________________
# def intialize_new_optimizer(model):
#     optimizer_class = DPOptimizer_Dice
#     optimizer_params = {
#         'optimizer': torch.optim.SGD(model.parameters(),learning_rate),
#         'noise_multiplier': 1.0,
#         'max_grad_norm': 0.5,
#         'expected_batch_size': 64,
#         'error_max_grad_norm': 1.0
#     }
#     return optimizer_class(**optimizer_params)




# if __name__ == "__main__":
#     delta = 1e-5
#     learning_rate = 0.05 # Learning rate for training
#     noise_multiplier = 2 # Ratio of the standard deviation to the clipping norm
#     clip_bound = 1 # Clipping norm
#     sample_rate = 0.03 # Batch size as a fraction of full data size
#     num_epochs = 1 # Number of epochs
#     torch.manual_seed(472368)

#     model_type = 'cnn'
#     # Load data
#     data_module = Data()
#     train_loader, test_loader,classes = data_module.cifar10(model_type)



#     # define the new algorithm:
#     # assume the Dice algorithm is the new one
#     # optimizers_models_dic =  intialize_models_optimizers(model_type,intialize_new_optimizer)

#     # benchmark
#     benchmark_module = Benchmark(model_type,intialize_new_optimizer)
#     benchmark_module.benchmark(train_loader, test_loader, learning_rate, sample_rate, noise_multiplier, clip_bound, delta, num_epochs)



