from abc import ABC, abstractmethod
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms

# Abstraction for Model Factories
class ModelFactory(ABC):
    @abstractmethod
    def create_model(self, num_classes: int, input_dims: tuple) -> nn.Module:
        """Creates and returns a model."""
        pass

    @abstractmethod
    def get_transform(self):
        """Returns the required transformations for the model."""
        pass

# VisionTransformer Implementation
class VisionTransformerFactory(ModelFactory):
    def create_model(self, num_classes: int, input_dims: tuple) -> nn.Module:
        from transformers import ViTModel
        class VisionTransformerNet(nn.Module):
            def __init__(self, num_classes):
                super(VisionTransformerNet, self).__init__()
                self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
                self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

            def forward(self, pixel_values):
                outputs = self.vit(pixel_values=pixel_values)
                pooled_output = outputs.pooler_output
                logits = self.classifier(pooled_output)
                return logits

        return VisionTransformerNet(num_classes)

    def get_transform(self):
        """Transformations required for Vision Transformer."""
        return transforms.Compose([
            transforms.Resize((224, 224))  # Fixed input size for ViT
        ])

# CNNNet Implementation
class CNNNet(nn.Module):
    def __init__(self, num_classes, input_dims):
        super(CNNNet, self).__init__()
        self.H, self.W, self.channels = input_dims
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

class CNNNetFactory(ModelFactory):
    def create_model(self, num_classes: int, input_dims: tuple) -> nn.Module:
        return CNNNet(num_classes, input_dims)

    def get_transform(self):
        """No additional transformations required for CNN."""
        return None

# SimpleCNN3c3d Implementation
class SimpleCNN3c3d(nn.Module):
    def __init__(self, num_classes, input_dims):
        super(SimpleCNN3c3d, self).__init__()
        self.H, self.W, self.channels = input_dims
        self.conv1 = nn.Conv2d(self.channels, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)

        fc_dims_H = np.floor(np.floor(np.floor((self.H - 2) / 2 - 2) / 2 - 2) / 2).astype(int)
        fc_dims_W = np.floor(np.floor(np.floor((self.W - 2) / 2 - 2) / 2 - 2) / 2).astype(int)
        self.fc1 = nn.Linear(128 * fc_dims_H * fc_dims_W, 256)
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

class SimpleCNN3c3dFactory(ModelFactory):
    def create_model(self, num_classes: int, input_dims: tuple) -> nn.Module:
        return SimpleCNN3c3d(num_classes, input_dims)

    def get_transform(self):
        """No additional transformations required for SimpleCNN3c3d."""
        return None

# SimpleCNN2c2d Implementation
class SimpleCNN2c2d(nn.Module):
    def __init__(self, num_classes, input_dims):
        super(SimpleCNN2c2d, self).__init__()
        self.H, self.W, self.channels = input_dims
        self.conv1 = nn.Conv2d(self.channels, 32, kernel_size=3)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        fc_dims_H = np.floor(np.floor((self.H - 2) / 2 - 2) / 2 ).astype(int)
        fc_dims_W = np.floor(np.floor((self.W - 2) / 2 - 2) / 2 ).astype(int)
        self.fc1 = nn.Linear(64 * fc_dims_H * fc_dims_W, 128) 
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class SimpleCNN2c2dFactory(ModelFactory):
    def create_model(self, num_classes: int, input_dims: tuple) -> nn.Module:
        return SimpleCNN2c2d(num_classes, input_dims)

    def get_transform(self):
        """No additional transformations required for SimpleCNN2c2d."""
        return None

# ConvVAE Implementation
class ConvVAE(nn.Module):
    def __init__(self, input_dims, latent_dim):
        super(ConvVAE, self).__init__()
        self.H, self.W, self.channels = input_dims

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Calculate the size of the flattened feature map
        self.fc_dims_H = np.floor((self.H + 2*1 - 3) / 2 + 1).astype(int)
        self.fc_dims_H  = np.floor((self.fc_dims_H  + 2*1 - 3) / 2 + 1).astype(int)
        self.fc_dims_H  = np.floor((self.fc_dims_H  + 2*1 - 3) / 2 + 1).astype(int)
        self.fc_dims_H  = np.floor((self.fc_dims_H  + 2*1 - 3) / 2 + 1).astype(int)

        self.fc_dims_W = np.floor((self.W + 2*1 - 3) / 2 + 1).astype(int)
        self.fc_dims_W  = np.floor((self.fc_dims_W  + 2*1 - 3) / 2 + 1).astype(int)
        self.fc_dims_W  = np.floor((self.fc_dims_W  + 2*1 - 3) / 2 + 1).astype(int)
        self.fc_dims_W  = np.floor((self.fc_dims_W  + 2*1 - 3) / 2 + 1).astype(int)


        self.fc1 = nn.Linear(256 * self.fc_dims_H  * self.fc_dims_W , 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)


        # Decoder
        self.fc2 = nn.Linear(latent_dim, 512)
        self.fc3 = nn.Linear(512, 256 * self.fc_dims_H  * self.fc_dims_W )
        #H_out =(H_in −1)×stride−2×padding+kernel_size+output_padding
        # this was done especificatlly for minset, if a new dataset was added, thing might need to change
        # if input changes from 28 to 32, just change the output_padding from zero to in second ConvTranspose2d
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
        nn.ConvTranspose2d(32, self.channels, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.fc2(z))
        x = F.relu(self.fc3(x))
        x = x.view(x.size(0), 256, self.fc_dims_H , self.fc_dims_W )
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

class ConvVAEFactory(ModelFactory):
    def __init__(self, latent_dim: int):
        self.latent_dim = latent_dim

    def create_model(self, num_classes: int, input_dims: tuple) -> nn.Module:
        return ConvVAE(input_dims, self.latent_dim)

    def get_transform(self):
        """No additional transformations required for ConvVAE."""
        return None