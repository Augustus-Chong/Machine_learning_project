import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from PIL import Image 
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report
import time

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20
DOWNLOAD_ROOT = './mnist_data'
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10
CUSTOM_IMAGE_PATH = 'custom_digit.png' 
MODEL_SAVE_PATH = 'mnist_saves/resnet_model.pth'
HIDDEN_SIZE = 128
NUM_RESIDUAL_BLOCKS = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        identity = x
        # Finding F(x)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        # Adding x, Output = F(x) + x
        out += identity 

        # Final activation
        out = self.relu(out)
        return out
    
class MinimalResNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_blocks):
        super().__init__()

        #Inital projection layer, (Input 784 -> Hidden Size)
        self.initial_layer = nn.Linear(input_size, hidden_size)

        # Stack of Residual Blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_size) for _ in range(num_blocks)])

        # Final classification layer (Hidden Size -> Output 10)
        self.final_layer = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.initial_layer(x))
        x = self.res_blocks(x)
        out = self.final_layer(x)
        return out

if __name__ == '__main__':
    #Get dataloaders
    train_loader, test_loader = get_data_loaders(DOWNLOAD_ROOT, BATCH_SIZE)
    #Initialize model
    model = MinimalResNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, NUM_RESIDUAL_BLOCKS).to(device)

    #load saved weights
    is_loaded = load_model(model, MODEL_SAVE_PATH, device)

    if not is_loaded:
        #Define loss function
        criterion  = nn.CrossEntropyLoss()
        #Define optimizer
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
       
        start_time = time.time()
        #Train the model
        loss_data = train_model(model, train_loader, criterion, optimizer, EPOCHS, device)

        end_time = time.time()
        training_duration = end_time - start_time
        
        print(f"\nTraining Complete in {training_duration:.2f} seconds.")

        save_model(model, MODEL_SAVE_PATH)
        #Evaluate model
        evaluate_model(model, test_loader, device)

        #visualing loss
        plot_loss(loss_data, window=20)
    else:
        print("\nSkipping training as weights were loaded.")
        evaluate_model(model, test_loader, device)

    predict_custom_image(model, CUSTOM_IMAGE_PATH, device)
