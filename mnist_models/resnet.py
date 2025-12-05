import torch
import torch.nn as nn
from Model_Base import run_model
import torch.nn.functional as F


BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 15
DOWNLOAD_ROOT = './mnist_data'
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10
CUSTOM_IMAGE_PATH = 'custom_digit.png' 
MODEL_SAVE_PATH = 'resnet_model3_momentum'
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
    
class ProperResidualBlock(nn.Module):
    """
    A standard ResNet block with Convolutional layers.
    Structure: Conv -> BN -> ReLU -> Conv -> BN -> Add Input -> ReLU
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # First Convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second Convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection (Identity Mapping)
        # If the input size (stride != 1) or channels change, we need to adapt x 
        # to match the output shape so we can add them.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # The Magic "Skip Connection": Adding the original input (x) to the output
        out += self.shortcut(x)
        
        out = F.relu(out)
        return out

class ConvResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Initial Layer: Input (1x28x28) -> Features (32x28x28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Residual Layers
        # Block 1: 32 channels, size stays 28x28
        self.layer1 = ProperResidualBlock(32, 32, stride=1)
        # Block 2: 64 channels, size shrinks to 14x14 (stride=2)
        self.layer2 = ProperResidualBlock(32, 64, stride=2)
        # Block 3: 128 channels, size shrinks to 7x7 (stride=2)
        self.layer3 = ProperResidualBlock(64, 128, stride=2)
        
        # Final Classification
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # NOTE: No flattening here! We keep the image dimensions.
        
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1) # Flatten only for the final classification
        out = self.fc(out)
        return out

if __name__ == '__main__':
    #Initialize model
    model = ConvResNet(NUM_CLASSES).to(device)
    #model = MinimalResNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, NUM_RESIDUAL_BLOCKS).to(device)
    run_model(model, BATCH_SIZE, EPOCHS, LEARNING_RATE, NUM_CLASSES, DOWNLOAD_ROOT, MODEL_SAVE_PATH, device, CUSTOM_IMAGE_PATH)


