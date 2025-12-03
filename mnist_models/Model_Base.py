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

def train_model(model, train_loader, criterion, optimizer, epochs, device):
    start_time = time.time()
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
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"\nTraining Complete in {training_duration:.2f} seconds.")
    return loss_history

def evaluate_model(model, test_loader, device):
    """Evaluates the model on the test dataset and prints detailed metrics."""
    model.eval() 
    y_true = [] # To store all true labels
    y_pred = [] # To store all predicted labels
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Store results for classification_report
            y_true.extend(labels.cpu().numpy()) # Move to CPU and convert to NumPy array
            y_pred.extend(predicted.cpu().numpy())
            
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / total_samples
    print(f'\n--- Evaluation Results ---')
    print(f'Overall Accuracy: {100 * accuracy:.2f}%')
    
    # Generate and print the comprehensive classification report
    print("\nDetailed Classification Report:")
    target_names = [str(i) for i in range(NUM_CLASSES)]
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    return accuracy

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

def predict_custom_image(model, image_path, device):
    """Loads a custom MNIST-like image and applies only necessary scaling and normalization."""
    if not os.path.exists(image_path):
        print(f"\n--- ERROR: Custom image '{image_path}' not found! ---")
        return

    print(f"\n--- Predicting Custom Image: {image_path} ---")
    
    try:
        # Load and convert to grayscale
        image = Image.open(image_path).convert('L') 
        
        # Define the simplified transformation pipeline
        simple_transform = transforms.Compose([
            transforms.Resize((28, 28)), # Safety resize to 28x28
            transforms.ToTensor(),       # Converts to Tensor and scales 0-1
        ])
        
        image_tensor = simple_transform(image).unsqueeze(0)
        
    except Exception as e:
        print(f"Failed to preprocess or predict custom image: {e}")
        return

    # --- Critical Inversion and Normalization ---
    # We keep the inversion check as a final safety measure against black/white reversal
    if image_tensor.mean() < 0.5:
        image_tensor = 1 - image_tensor 

    mean_tensor = torch.tensor([0.1307]).view(1, 1, 1, 1)
    std_tensor = torch.tensor([0.3081]).view(1, 1, 1, 1)
    image_tensor = (image_tensor - mean_tensor) / std_tensor
    
    image_tensor = image_tensor.to(device)
    
    model.eval()
    
    with torch.no_grad():
        output = model(image_tensor)
    
    probabilities = F.softmax(output, dim=1) 
    
    _, predicted_class_tensor = torch.max(output, 1)
    predicted_class = predicted_class_tensor.item()
    
    confidence = probabilities[0][predicted_class].item() * 100
    
    print(f"Model prediction: The digit is {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    
    return predicted_class

def plot_loss(loss_data, window=10):
    """
    Plots the raw loss and a smoothed version for clearer convergence analysis.
    The plot also focuses on the later training steps.
    """
    
    # 1. Calculate Moving Average for Smoothing
    loss_np = np.array(loss_data)
    # Pads the start with the first value so the smoothed curve starts at index 0
    padded_loss = np.pad(loss_np, (window-1, 0), mode='edge')
    smoothed_loss = np.convolve(padded_loss, np.ones(window)/window, mode='valid')
    
    # 2. Define Zoom Range
    # Start plotting at 30% of the total training steps to skip the steep initial drop
    zoom_start_index = int(len(loss_data) * 0.3) 
    
    
    plt.figure(figsize=(12, 6))

    # Plot 1: Smoothed Loss (Main Trend)
    plt.plot(smoothed_loss, label=f'Smoothed Loss (Window={window})', color='darkorange', linewidth=2)
    
    # Plot 2: Raw Loss (Volatility check)
    plt.plot(loss_data, label='Raw Loss (Volatility)', color='skyblue', alpha=0.3)
    
    # Optional: Plot 3: Zoomed in section (for extra clarity)
    plt.plot(np.arange(zoom_start_index, len(loss_data)), 
             smoothed_loss[zoom_start_index:], 
             label='Zoomed Trend', color='red', linestyle='--')
    
    plt.grid(axis="both", linewidth=1, color="lightgrey", linestyle="dashed")
    plt.gca().set_xticks(np.arange(0, 1.0, 0.05))


    plt.title("Loss over Training Steps: Trend vs. Volatility")
    plt.xlabel(f"Training Step (x100 batches) [Total Steps: {len(loss_data)}]")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.6)
    plt.show()