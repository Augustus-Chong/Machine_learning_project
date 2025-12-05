import torch
import torch.nn as nn
import torch.optim as optim
from Model_Base import run_model

BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 30
DOWNLOAD_ROOT = './mnist_data'
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10
CUSTOM_IMAGE_PATH = 'custom_digit.png' 
MODEL_SAVE_PATH = 'logistic_model2'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1) #flatten image into vector
        out = self.linear(x)
        return out


if __name__ == '__main__':
    #Initialize model
    model = LogisticRegression(INPUT_SIZE, NUM_CLASSES).to(device)
    run_model(model, BATCH_SIZE, EPOCHS, LEARNING_RATE, NUM_CLASSES, DOWNLOAD_ROOT, MODEL_SAVE_PATH, device, CUSTOM_IMAGE_PATH)
