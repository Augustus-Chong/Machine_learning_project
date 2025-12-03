import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from PIL import Image 
import torch.nn.functional as F

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 100
DOWNLOAD_ROOT = './mnist_data'
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10
CUSTOM_IMAGE_PATH = 'custom_digit.png' 
MODEL_SAVE_PATH = 'mnist_saves/resnet_model.pth'
HIDDEN_SIZE = 128
NUM_RESIDUAL_BLOCKS = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_data_loaders(root, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) 
    ])
    is_downloaded = os.path.isdir(root)

    train_dataset = datasets.MNIST( #Training dataset
        root=root, 
        train=True,         
        download=not is_downloaded, 
        transform=transform 
    )

    test_dataset = datasets.MNIST( #Test dataset
        root=root,
        train=False,        
        download=not is_downloaded, 
        transform=transform 
    )

    #dataloaders for batching and shuffing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
    
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


def train_model(model, train_loader, criterion, optimizer, epochs, device):
    model.train()
    loss_history = []
    for epoch in range(epochs):
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad() #flush out previous gradients + initialization grads
            outputs = model(images) #Forward Pass
            loss = criterion(outputs, labels) #compute loss
            loss.backward() #Backward pass, computes gradients
            optimizer.step() #update weights using the optimizer chosen

            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                loss_history.append(loss.item())
    return loss_history

def evaluate_model(model, test_loader, device):
    model.eval()
    total_correct = 0
    total_samples  =0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images) #forward pass
            _, predicted = torch.max(outputs.data, 1)

            total_samples += labels.size(0)
            total_correct += (predicted ==labels).sum().item()

    accuracy = total_correct / total_samples
    print(f'\nTest Accuracy: {100 * accuracy:.2f}%')
    return accuracy

def predict_custom_image(model, image_path, device):
    """Loads a single image, preprocesses it robustly, and makes a prediction."""
    if not os.path.exists(image_path):
        print(f"\n--- ERROR: Custom image '{image_path}' not found! ---")
        print("Please draw a digit, save it as a PNG file, and ensure it is in the same directory.")
        return

    print(f"\n--- Predicting Custom Image: {image_path} ---")
    
    try:
        # Load and convert to grayscale
        # We perform initial conversion here
        image = Image.open(image_path).convert('L') 
    except Exception as e:
        print(f"Failed to load image: {e}")
        return

    # Transformations for custom image: Resize to 28x28 and convert to Tensor
    custom_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        # Apply a strict binary threshold (0 or 255) to mimic crisp MNIST lines
        transforms.Lambda(lambda x: x.point(lambda p: 0 if p < 128 else 255)),
        transforms.ToTensor(),
    ])
    
    image_tensor = custom_transform(image).unsqueeze(0)
    
    # --- Critical Inversion and Normalization ---
    # The ToTensor conversion changes pixel range from 0-255 to 0.0-1.0
    
    # Check if image needs to be inverted (if background is dark, mean will be low)
    # MNIST digits are represented by high values (close to 1.0)
    if image_tensor.mean() < 0.5:
        # Invert: white becomes black, black becomes white
        image_tensor = 1 - image_tensor 

    # Apply the standard MNIST normalization after potential inversion
    # Normalization: (x - mean) / std
    mean_tensor = torch.tensor([0.1307]).view(1, 1, 1, 1)
    std_tensor = torch.tensor([0.3081]).view(1, 1, 1, 1)
    image_tensor = (image_tensor - mean_tensor) / std_tensor
    
    image_tensor = image_tensor.to(device)
    
    model.eval()
    
    with torch.no_grad():
        output = model(image_tensor)
    
    # Calculate probabilities using Softmax
    probabilities = F.softmax(output, dim=1) 
    
    _, predicted_class_tensor = torch.max(output, 1)
    predicted_class = predicted_class_tensor.item()
    
    confidence = probabilities[0][predicted_class].item() * 100
    
    print(f"Model prediction: The digit is {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    
    return predicted_class

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"\nModel weights saved to {path}")

def load_model(model, path, device):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        print(f"\nModel weights successfully loaded from {path}")
        return True
    return False

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
       

        #Train the model
        loss_data = train_model(model, train_loader, criterion, optimizer, EPOCHS, device)
        save_model(model, MODEL_SAVE_PATH)
        #Evaluate model
        evaluate_model(model, test_loader, device)

        #visualing loss
        plt.figure(figsize=(10, 5))
        plt.plot(loss_data)
        plt.title("Loss over Training Steps")
        plt.xlabel("Training Step (x100 batches)")
        plt.ylabel("Loss")
        plt.show()
    else:
        print("\nSkipping training as weights were loaded.")
        evaluate_model(model, test_loader, device)

    predict_custom_image(model, CUSTOM_IMAGE_PATH, device)
