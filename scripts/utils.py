# Some useful function

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.utils.data import Dataset




# set up the device to use GPU if available
def use_GPU():
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
    device = torch.device("cuda:0" if train_on_gpu else "cpu")
    print(device)

    return device

## Confusion matrix and predictions

def get_all_predictions(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            preds = model(images)
            all_preds.append(preds.detach())  # Detach to avoid tracking gradient
    return torch.cat(all_preds, dim=0).to('cpu')

def print_confusion_matrix_no_diag(model, testset, name_of_classes):
    # no shuflfe testset so using a loader appropriately, i could use testset directly but loader is more memory efficient
    loader = torch.utils.data.DataLoader(testset,
                                         batch_size = 4,
                                         shuffle = False,
                                         num_workers = 2)
    
    # Consistent device usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    predictions = get_all_predictions(model, loader)
    predicted_labels = predictions.argmax(dim=1)
    
    # Collect true labels and ensure they are on the same device as predictions
    true_labels = torch.cat([y for _, y in loader], dim=0).to(device)
    
    # Calculate confusion matrix using numpy as it is not dependent on the device
    cm = confusion_matrix(true_labels.cpu().numpy(), predicted_labels.cpu().numpy())
    n = cm.shape[0]
    cm *= (np.ones([n,n], dtype=np.int64 ) - np.identity(n, dtype=np.int64))
    
    # Visualization using matplotlib and seaborn
    plt.figure(figsize=(13, 10))
    sns.heatmap(cm, 
                annot=True, 
                fmt="d", 
                cmap='viridis', 
                xticklabels=name_of_classes, 
                yticklabels=name_of_classes,
                robust=True)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('confucio.png')
    plt.show()

def print_confusion_matrix(model, testset, name_of_classes):
    # no shuflfe testset so using a loader appropriately, i could use testset directly but loader is more memory efficient
    loader = torch.utils.data.DataLoader(testset,
                                         batch_size = 4,
                                         shuffle = False,
                                         num_workers = 2)
    
    # Consistent device usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    predictions = get_all_predictions(model, loader)
    predicted_labels = predictions.argmax(dim=1)
    
    # Collect true labels and ensure they are on the same device as predictions
    true_labels = torch.cat([y for _, y in loader], dim=0).to(device)
    
    # Calculate confusion matrix using numpy as it is not dependent on the device
    cm = confusion_matrix(true_labels.cpu().numpy(), predicted_labels.cpu().numpy())
    
    # Visualization using matplotlib and seaborn
    plt.figure(figsize=(13, 10))
    sns.heatmap(cm, 
                annot=True, 
                fmt="d", 
                cmap='viridis', 
                xticklabels=name_of_classes, 
                yticklabels=name_of_classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('confucio.png')
    plt.show()

