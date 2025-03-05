import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
import numpy as np
# class VisionTransformerNet(nn.Module):
#     def __init__(self, num_classes):
#         super(VisionTransformerNet, self).__init__()
#         self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
#         self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

#     def forward(self, pixel_values):
#         outputs = self.vit(pixel_values=pixel_values)
#         pooled_output = outputs.pooler_output
#         logits = self.classifier(pooled_output)
#         return logits

class CNNNet(nn.Module):
    def __init__(self, num_classes, input_dims):
        super(CNNNet, self).__init__()
        self.H, self.W,self.channels = input_dims
        self.conv1 = nn.Conv2d(self.channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        fc_dims_H = np.floor(np.floor((self.H - 4) / 2 - 4) / 2).astype(int)
        fc_dims_W = np.floor(np.floor((self.W - 4) / 2 - 4) / 2).astype(int)
        self.fc1 = nn.Linear(16 * fc_dims_H * fc_dims_W, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # print()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class SimpleCNN3c3d(nn.Module):
    def __init__(self, num_classes, input_dims):
        super(SimpleCNN3c3d, self).__init__()
        self.H, self.W,self.channels = input_dims
        self.conv1 = nn.Conv2d(self.channels, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)

        fc_dims_H = np.floor(np.floor(np.floor((self.H - 2) / 2 - 2) / 2 - 2) / 2).astype(int)
        fc_dims_W = np.floor(np.floor(np.floor((self.W - 2) / 2 - 2) / 2 - 2) / 2).astype(int)
        self.fc1 = nn.Linear(128*fc_dims_H*fc_dims_W, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class SimpleCNN2c2d(nn.Module):
    def __init__(self, num_classes, input_dims):
        super(SimpleCNN2c2d, self).__init__()
        self.H, self.W,self.channels = input_dims
        self.conv1 = nn.Conv2d(self.channels, 32, kernel_size=3)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        fc_dims_H = np.floor(np.floor((self.H - 2) / 2 - 2) / 2 ).astype(int)
        fc_dims_W = np.floor(np.floor((self.W - 2) / 2 - 2) / 2 ).astype(int)
        self.fc1 = nn.Linear(64*fc_dims_H*fc_dims_W, 128) 
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)