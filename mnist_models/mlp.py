import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Model_Base import run_model
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 15
DOWNLOAD_ROOT = './mnist_data'
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10
CUSTOM_IMAGE_PATH = 'custom_digit.png' 
MODEL_SAVE_PATH = 'mnist_saves/mlp_model1.pth'
HIDDEN_SIZE = 128


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        
        # 1. Input Layer to Hidden Layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 2. Hidden Layer to Output Layer
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1) 
        # First Linear transformation + Activation Function (ReLU)
        x = F.relu(self.fc1(x))
        
        # Second Linear transformation to output scores
        out = self.fc2(x)
        return out

if __name__ == '__main__':
    #Initialize model
    model = MLP(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
    run_model(model, BATCH_SIZE, EPOCHS, LEARNING_RATE, NUM_CLASSES, DOWNLOAD_ROOT, MODEL_SAVE_PATH, device, CUSTOM_IMAGE_PATH)
