import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import dataloader
import matplotlib.pyplot as plt
import os

BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 5
DOWNLOAD_ROOT = './mnist_data'
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")