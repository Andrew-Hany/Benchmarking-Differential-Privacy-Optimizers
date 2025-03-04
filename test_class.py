import torch
import numpy as np
from tqdm import tqdm
class Testing():
    @staticmethod   
    def prediction_function(output):
        # Assuming output is a tensor of shape (batch_size, num_classes)
        return torch.argmax(output, dim=1)
    @staticmethod
    def test(model: torch.nn.Module,criterion, 
             test_loader: torch.utils.data.DataLoader,
             device: torch.device,
            prediction_function: callable = None):
        if prediction_function is None:
            raise ValueError("Please provide prediction_function.")

        with torch.no_grad():
            losses = []
            accuracies = []
            total_size = 0
            for x, y in tqdm(test_loader, desc=f'Testing'):
                x, y = x.to(device), y.to(device)
                total_size += len(y)
                out = model(x)
                loss = criterion(out, y)
                preds = prediction_function(out)
                corrects = torch.tensor(torch.sum(preds == y).item())
                losses.append(loss)
                accuracies.append(corrects)

            average_loss = np.array(loss).sum() / total_size
            total_accuracy = np.array(accuracies).sum() / total_size
            return average_loss, total_accuracy