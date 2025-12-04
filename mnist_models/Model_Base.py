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
from sklearn.metrics import classification_report, confusion_matrix
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

def evaluate_model(model, test_loader, device, NUM_CLASSES):
    """Evaluates the model on the test dataset and prints detailed metrics, including per-class accuracy."""
    model.eval() 
    y_true = [] 
    y_pred = [] 
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / total_samples
    print(f'\n--- Evaluation Results ---')
    print(f'Overall Accuracy: {100 * accuracy:.2f}%')
    
    # --- Detailed Per-Class Metrics ---
    
    # 1. Get the Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # 2. Extract True Positives (TP) from the diagonal
    correct_per_class = cm.diagonal()
    
    # 3. Extract Total Samples (Support) for each class
    total_per_class = cm.sum(axis=1) # Sum of actual rows in the confusion matrix
    
    # 4. Calculate Per-Class Accuracy (TP / Total Samples)
    class_accuracy = correct_per_class / total_per_class
    
    # 5. Print custom accuracy table
    print("\nPer-Class Accuracy:")
    print("---------------------------------")
    print(" Class | Total Samples | Accuracy ")
    print("---------------------------------")
    
    for i in range(NUM_CLASSES):
        print(f"   {i}   |     {total_per_class[i]:<10} | {class_accuracy[i]*100:.2f}%")
        
    print("---------------------------------")
    
    # 6. Generate and print the comprehensive classification report
    print("\nDetailed Classification Report (Precision, Recall, F1-Score):")
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
    """
    Loads a custom image, applies the EXACT same preprocessing as MNIST training
    (Resize -> ToTensor -> Inversion Check -> Normalize), and predicts with confidence.
    """
    if not os.path.exists(image_path):
        print(f"\n--- ERROR: Custom image '{image_path}' not found! ---")
        return

    print(f"\n--- Predicting Custom Image: {image_path} ---")
    
    try:
        # 1. Load and convert to grayscale
        image = Image.open(image_path).convert('L') 
        
        # 2. Resize and Convert to Tensor (Scales pixels to 0.0 - 1.0)
        simple_transform = transforms.Compose([
            transforms.Resize((28, 28)), 
            transforms.ToTensor(), 
        ])
        
        # Add batch dimension: Shape becomes (1, 1, 28, 28)
        image_tensor = simple_transform(image).unsqueeze(0)
        
    except Exception as e:
        print(f"Failed to preprocess or predict custom image: {e}")
        return

    # 3. Inversion Check 
    # (Ensures digit is white/high-value and background is black/low-value)
    #if image_tensor.mean() < 0.5:
    #    image_tensor = 1 - image_tensor 

    # 4. CRITICAL: Normalization
    # This aligns the custom image distribution with the MNIST training data
    # (shifting the range from [0, 1] to approx [-0.42, 2.82])
    mean_tensor = torch.tensor([0.1307]).view(1, 1, 1, 1)
    std_tensor = torch.tensor([0.3081]).view(1, 1, 1, 1)
    
    image_tensor = (image_tensor - mean_tensor) / std_tensor
    
    # 5. Prediction
    image_tensor = image_tensor.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(image_tensor)
    
    # 6. Interpret Results
    probabilities = F.softmax(output, dim=1) 
    
    _, predicted_class_tensor = torch.max(output, 1)
    predicted_class = predicted_class_tensor.item()
    
    confidence = probabilities[0][predicted_class].item() * 100
    
    # Extract probabilities as a flat list
    all_confidences = probabilities.squeeze().cpu().numpy()
    
    print(f"\nPrediction Result:")
    print(f"-> Predicted Digit: {predicted_class}")
    print(f"-> Highest Confidence: {confidence:.2f}%")
    print("-" * 30)
    print("Class | Confidence")
    print("-" * 30)

    # Print the confidence for all 10 classes
    for i, conf in enumerate(all_confidences):
        marker = " <--- WINNER" if i == predicted_class else ""
        print(f"  {i}   | {conf * 100:.2f}%{marker}")
    print("-" * 30)
    
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

    plt.title("Loss over Training Steps: Trend vs. Volatility")
    plt.xlabel(f"Training Step (x100 batches) [Total Steps: {len(loss_data)}]")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.6)
    plt.show()

def run_model(model, BATCH_SIZE, EPOCHS, LEARNING_RATE, NUMBER_CLASSES, DOWNLOAD_ROOT, MODEL_SAVE_PATH, device, CUSTOM_IMAGE_PATH):
    #Get dataloaders
    train_loader, test_loader = get_data_loaders(DOWNLOAD_ROOT, BATCH_SIZE)
    
    #load saved weights
    is_loaded = load_model(model, MODEL_SAVE_PATH, device)

    if not is_loaded:
        #Define loss function
        criterion  = nn.CrossEntropyLoss()
        #Define optimizer
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
       
        #Train the model
        loss_data = train_model(model, train_loader, criterion, optimizer, EPOCHS, device)

        save_model(model, MODEL_SAVE_PATH)
        #Evaluate model
        evaluate_model(model, test_loader, device, NUMBER_CLASSES)

        #visualing loss
        plot_loss(loss_data, window=20)
    else:
        print("\nSkipping training as weights were loaded.")
        evaluate_model(model, test_loader, device, NUMBER_CLASSES)

    predict_custom_image(model, CUSTOM_IMAGE_PATH, device)

def load_model2(model, MODEL_PATH, device):
    """Loads the trained model weights."""
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        return model
    raise FileNotFoundError(f"ResNet model weights not found at {MODEL_PATH}. Please train and save it first.")

def predict(model, input_tensor):
    """Predicts using the PyTorch model."""
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        _, predicted_class_tensor = torch.max(output, 1)
        
        predicted_class = predicted_class_tensor.item()
        confidence = probabilities[0][predicted_class].item() * 100
        
        print(f"MODEL: ")
        print(f"  Prediction: {predicted_class}")
        print(f"  Confidence: {confidence:.2f}%")
        print("-" * 50)