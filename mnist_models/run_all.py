import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import joblib
from PIL import Image, ImageOps 
from torchvision import transforms
import matplotlib.pyplot as plt

from resnet import MinimalResNet
from mlp import MLP
from logistic import LogisticRegression
from Model_Base import load_model2, predict

# --- CONFIGURATION ---
CUSTOM_IMAGE_PATH = 'custom_digit.png' 
MODEL_PATH_RESNET = 'mnist_saves/resnet_model.pth' 
MODEL_PATH_MLP = 'mnist_saves/mlp_model.pth' 
MODEL_PATH_LOGISTIC = 'mnist_saves/logistic_model.pth' 

# Architectural parameters (must match training)
INPUT_SIZE = 28*28
HIDDEN_SIZE = 128
NUM_CLASSES = 10     
NUM_RESIDUAL_BLOCKS = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- CUSTOM IMAGE PREPROCESSING (Must be shared across all models) ---

def preprocess_custom_image(image_path):
    """
    Loads custom image, crops to bounding box, scales to 20x20 (MNIST standard),
    centers it in a 28x28 box, and applies normalization.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Custom image '{image_path}' not found.")

    # 1. Load Image
    image = Image.open(image_path).convert('L') 
    
    # 2. Thresholding (Make it binary 0 or 255 to find bounding box easily)
    # This removes faint noise from the background
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
            
        # Resize the cropped digit
        crop_resized = crop.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create a blank 28x28 black canvas
        final_image = Image.new('L', (28, 28), 0)
        
        # Paste the resized digit into the center of the canvas
        paste_x = (28 - new_width) // 2
        paste_y = (28 - new_height) // 2
        final_image.paste(crop_resized, (paste_x, paste_y))
    else:
        # If the image is empty/black, just return the empty black image
        final_image = image.resize((28, 28))

    # 4. Convert to Tensor (0.0 - 1.0)
    to_tensor = transforms.ToTensor()
    image_tensor_1d = to_tensor(final_image)
    
    # --- VISUALIZATION DATA ---
    raw_plot_data = image_tensor_1d.squeeze().numpy()
    
    # 5. Skip Inversion Check (We handled black/white logic in Step 2)
    image_tensor_processed = image_tensor_1d 

    # --- VISUALIZATION DATA ---
    inverted_plot_data = image_tensor_processed.squeeze().numpy()
    
    # 6. Prepare Outputs
    # Scikit-learn (NumPy, Flat)
    numpy_flat = image_tensor_processed.squeeze().numpy().reshape(1, INPUT_SIZE) 
    
    # PyTorch (Tensor, Normalized)
    tensor_input = image_tensor_processed.unsqueeze(0)
    
    # Apply Standard MNIST Normalization
    mean_tensor = torch.tensor([0.1307]).view(1, 1, 1, 1)
    std_tensor = torch.tensor([0.3081]).view(1, 1, 1, 1)
    normalized_tensor = (tensor_input - mean_tensor) / std_tensor
    
    # --- VISUALIZATION DATA ---
    final_plot_data = normalized_tensor.cpu().squeeze().numpy() * std_tensor.item() + mean_tensor.item()
    
    # --- PLOTTING ---
    try:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(raw_plot_data, cmap='gray')
        plt.title(f"1. Smart Center (20px)\nMean: {raw_plot_data.mean():.3f}")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(inverted_plot_data, cmap='gray')
        plt.title(f"2. Processing\nMean: {inverted_plot_data.mean():.3f}")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(final_plot_data, cmap='gray')
        plt.title(f"3. Final Input (Normalized)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[Plotting Error]: {e}")
        
    return normalized_tensor.to(device)

if __name__ == '__main__':
    try:
        # 1. Preprocess the custom image once
        models_input = preprocess_custom_image(CUSTOM_IMAGE_PATH)
        
        # 2. Load Models
        Rmodel = MinimalResNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, NUM_RESIDUAL_BLOCKS).to(device)
        resnet_model = load_model2(Rmodel, MODEL_PATH_RESNET, device)

        Mmodel = MLP(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
        mlp_model = load_model2(Mmodel, MODEL_PATH_MLP, device)

        Lmodel = LogisticRegression(INPUT_SIZE, NUM_CLASSES).to(device)
        logistic_model = load_model2(Lmodel, MODEL_PATH_LOGISTIC, device)

        print("\n" + "="*50)
        print(f"| RUNNING MULTI-MODEL PREDICTION ON: {CUSTOM_IMAGE_PATH}")
        print("="*50)

        # 3. Predict with ResNet (Deep Learning)
        predict(resnet_model, models_input)
        predict(mlp_model, models_input)
        predict(logistic_model, models_input)
       
        
    except FileNotFoundError as e:
        print("\n!!! SETUP ERROR !!!")
        print(e)
        print("Please ensure you have run the appropriate training scripts (e.g., mnist_knn_classifier.py and the PyTorch file) to create the saved model files.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")