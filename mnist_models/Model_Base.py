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

def evaluate_model(model, test_loader, device, NUM_CLASSES):
    """Evaluates the model and plots a high-contrast accuracy chart."""
    model.eval() 
    y_true = [] 
    y_pred = [] 
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # --- Metrics ---
    cm = confusion_matrix(y_true, y_pred)
    correct_per_class = cm.diagonal()
    total_per_class = cm.sum(axis=1)
    class_accuracy = correct_per_class / total_per_class
    overall_accuracy = np.mean(class_accuracy) # Macro average for coloring logic

    print(f'\nOverall Accuracy: {100 * overall_accuracy:.4f}%')
    target_names = [str(i) for i in range(NUM_CLASSES)]
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    # --- IMPROVED PLOTTING ---
    plt.figure(figsize=(10, 6))
    
    # 1. Define Colors based on performance relative to average
    colors = []
    for acc in class_accuracy:
        if acc < overall_accuracy - 0.02: # significantly below average
            colors.append('salmon') # Red-ish
        elif acc < overall_accuracy:      # slightly below
            colors.append('gold')   # Yellow
        else:
            colors.append('lightgreen') # Good
            
    bars = plt.bar(range(NUM_CLASSES), class_accuracy * 100, color=colors, edgecolor='black')
    
    plt.title(f'Accuracy per Digit Class (Avg: {overall_accuracy*100:.1f}%)')
    plt.xlabel('Digit Class')
    plt.ylabel('Accuracy (%)')
    plt.xticks(range(NUM_CLASSES))
    
    # 2. Zoom in the Y-Axis to show differences
    # Start 5% below the worst class, capped at 0
    lowest_acc = min(class_accuracy) * 100
    plt.ylim(max(0, lowest_acc - 5), 100.5) 
    
    # 3. Add Average Line
    plt.axhline(y=overall_accuracy*100, color='blue', linestyle='--', alpha=0.5, label='Average Accuracy')
    plt.legend()

    # Labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.show()
    
    return overall_accuracy

def predict_custom_image(model, image_path, device):
    """
    Loads a custom image, applies the STRICT MNIST protocol 
    (Crop -> Scale to 20x20 -> Center in 28x28 -> Normalize), 
    and predicts with confidence.
    """
    if not os.path.exists(image_path):
        print(f"\n--- ERROR: Custom image '{image_path}' not found! ---")
        return

    print(f"\n--- Predicting Custom Image: {image_path} ---")
    
    try:
        # 1. Load and convert to grayscale
        image = Image.open(image_path).convert('L') 
        
        # 2. Thresholding (Make it binary to find bounding box easily)
        # This forces the digit to be white (255) and background black (0)
        image = image.point(lambda p: 0 if p < 128 else 255)

        # 3. Smart Crop & Scale (The MNIST Protocol)
        # Find the bounding box of the non-zero (white) pixels
        bbox = image.getbbox()
        
        if bbox:
            # Crop the digit to the tightest rectangle
            crop = image.crop(bbox)
            
            # Calculate scale factor to fit into a 20x20 box (leaving 4px padding)
            width, height = crop.size
            if width > height:
                new_width = 20
                new_height = int(height * (20 / width))
            else:
                new_height = 20
                new_width = int(width * (20 / height))
                
            # Resize the cropped digit using high-quality resampling
            crop_resized = crop.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create a blank 28x28 black canvas
            final_image = Image.new('L', (28, 28), 0)
            
            # Paste the resized digit into the center of the canvas
            paste_x = (28 - new_width) // 2
            paste_y = (28 - new_height) // 2
            final_image.paste(crop_resized, (paste_x, paste_y))
        else:
            # If image is empty, just return a black 28x28 square
            final_image = image.resize((28, 28))

        # 4. Convert to Tensor (Scales pixels to 0.0 - 1.0)
        to_tensor = transforms.ToTensor()
        image_tensor = to_tensor(final_image).unsqueeze(0)
        
    except Exception as e:
        print(f"Failed to preprocess or predict custom image: {e}")
        return

    # 5. Inversion Check (Optional if you trust the thresholding step above)
    # Since we forced white-on-black in step 2, this is a safety net.
    if image_tensor.mean() < 0.5:
         pass # Already correct (mostly black background)
    else:
         image_tensor = 1 - image_tensor

    # 6. CRITICAL: Normalization
    # Matches the training data distribution [-0.42 to 2.82]
    mean_tensor = torch.tensor([0.1307]).view(1, 1, 1, 1)
    std_tensor = torch.tensor([0.3081]).view(1, 1, 1, 1)
    image_tensor = (image_tensor - mean_tensor) / std_tensor
    
    # 7. Prediction
    image_tensor = image_tensor.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(image_tensor)
    
    # 8. Interpret Results
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

def plot_loss(loss_data, epochs, window=10):
    """
    Plots the raw loss and a smoothed version with the X-axis scaled to Epochs.
    """
    
    # 1. Calculate Moving Average
    loss_np = np.array(loss_data)
    padded_loss = np.pad(loss_np, (window-1, 0), mode='edge')
    smoothed_loss = np.convolve(padded_loss, np.ones(window)/window, mode='valid')
    
    # 2. Create the Epochs Axis
    # Generate an array from 0 to total_epochs
    steps_per_epoch = len(loss_data) / epochs
    x_axis = np.linspace(0, epochs, len(loss_data))
    
    zoom_start_idx = int(len(loss_data) * 0.3) 
    
    plt.figure(figsize=(12, 6))

    # Plot 1: Smoothed Loss
    plt.plot(x_axis, smoothed_loss, label=f'Smoothed Loss (Window={window})', color='darkorange', linewidth=2)
    
    # Plot 2: Raw Loss
    plt.plot(x_axis, loss_data, label='Raw Loss (Volatility)', color='skyblue', alpha=0.3)
    
    # Optional: Plot 3: Zoomed Trend
    plt.plot(x_axis[zoom_start_idx:], 
             smoothed_loss[zoom_start_idx:], 
             label='Zoomed Trend', color='red', linestyle='--')
    
    plt.grid(axis="both", linewidth=1, color="lightgrey", linestyle="dashed")

    plt.title("Loss over Training Epochs")
    plt.xlabel("Epoch") # Changed label
    plt.ylabel("Loss")
    
    # Set X-ticks to show integers for every epoch
    plt.xticks(np.arange(0, epochs + 1, step=max(1, epochs//10)))
    plt.xlim(0, epochs)
    
    plt.legend()
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
        #
        #Train the model
        loss_data = train_model(model, train_loader, criterion, optimizer, EPOCHS, device)

        save_model(model, MODEL_SAVE_PATH)
        #Evaluate model
        evaluate_model(model, test_loader, device, NUMBER_CLASSES)

        #visualing loss
        plot_loss(loss_data, EPOCHS, window=20)
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