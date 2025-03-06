import torch
import numpy as np
from tqdm import tqdm
class Testing():
    @staticmethod   
    def prediction_function(output):
        # Assuming output is a tensor of shape (batch_size, num_classes)
        return torch.argmax(output, dim=1)
    @staticmethod
    def classification_test(model: torch.nn.Module,criterion, 
             test_loader: torch.utils.data.DataLoader,
             device: torch.device,
            prediction_function: callable = None):
        if prediction_function is None:
            raise ValueError("Please provide prediction_function.")

        model.eval()
        with torch.no_grad():
            losses = []
            accuracies = []
            total_size = 0
            for x, y in tqdm(test_loader, desc=f'Testing'):
                x, y = x.to(device), y.to(device)
                total_size += x.size(0)
                out = model(x)
                loss = criterion(out, y)
                preds = prediction_function(out)
                corrects = torch.tensor(torch.sum(preds == y).item())
                losses.append(loss.item()*x.size(0))  # Convert average batch loss to total batch loss
                                                  

                accuracies.append(corrects)

            average_loss = np.sum(losses) / total_size  # loss per sample =  average batches lossv/total_size
            total_accuracy = np.sum(accuracies) / total_size 

            return average_loss, total_accuracy

    @staticmethod
    def VAE_test(model: torch.nn.Module, criterion, 
                 test_loader: torch.utils.data.DataLoader,
                 device: torch.device):
        model.eval()
        with torch.no_grad():
            losses = []
            total_size = 0
            for x, _ in tqdm(test_loader, desc='Testing'):
                x = x.to(device)
                total_size += len(x)
                recon_x, mu, logvar = model(x)
                loss = criterion(recon_x, x, mu, logvar)
                losses.append(loss.item()*x.size(0)) 

            average_loss = np.sum(losses) / total_size 
            return average_loss

    @staticmethod
    def test(model_type, model: torch.nn.Module, criterion, 
             test_loader: torch.utils.data.DataLoader,
             device: torch.device,
             prediction_function: callable = None):
        if model_type.lower() == 'classification':
            return Testing.classification_test(model, criterion, test_loader, device, prediction_function)
        elif model_type.lower() == 'vae':
            return Testing.VAE_test(model, criterion, test_loader, device), None
        else:
            raise ValueError("Unsupported model type")


