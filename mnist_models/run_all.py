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
NUM_RESIDUAL_BLOCKS = 3 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- CUSTOM IMAGE PREPROCESSING (Must be shared across all models) ---

def preprocess_custom_image(image_path):
    """
    Loads pre-sized custom image (28x28), applies minimal scaling,
    and returns the normalized Tensor (for PyTorch models) and flat NumPy array.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Custom image '{image_path}' not found.")

    image = Image.open(image_path).convert('L') # Load as grayscale
    
    # 1. Minimal Transformation Pipeline (PIL to Tensor, 0-1 scaling)
    minimal_transform = transforms.Compose([
        transforms.Resize((28, 28)), 
        transforms.ToTensor(), # Scales pixels to 0.0 - 1.0
    ])
    
    # Shape: (1, 28, 28)
    image_tensor_1d = minimal_transform(image)
    
    # --- VISUALIZATION STEP 1: RAW INPUT ---
    raw_plot_data = image_tensor_1d.squeeze().numpy() 
    
    # 2. Skip Inversion Check (We trust the input is already white-on-black)
    image_tensor_processed = image_tensor_1d # Assign without modifying
    
    # --- VISUALIZATION STEP 2: NO INVERSION ---
    inverted_plot_data = image_tensor_processed.squeeze().numpy()
    
    # 3. Prepare Scikit-learn Output (NumPy array)
    numpy_flat = image_tensor_processed.squeeze().numpy().reshape(1, INPUT_SIZE) 
    
    # 4. Prepare PyTorch Output (Normalized Tensor)
    tensor_input = image_tensor_processed.unsqueeze(0)
    
    # Apply standard MNIST normalization
    mean_tensor = 0.1307
    std_tensor = 0.3081
    normalized_tensor = (tensor_input - mean_tensor) / std_tensor
    
    # --- VISUALIZATION STEP 3: FINAL NORMALIZED INPUT ---
    # Denormalize to display (x * std + mean)
    final_plot_data = normalized_tensor.cpu().squeeze().numpy() * std_tensor + mean_tensor
    
    
    # --- PLOTTING ---
    try:
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(raw_plot_data, cmap='gray')
        plt.title(f"1. Raw (0-1 Scale)\nMean: {raw_plot_data.mean():.3f}")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(inverted_plot_data, cmap='gray')
        plt.title(f"2. After Inversion Check\nMean: {inverted_plot_data.mean():.3f}")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(final_plot_data, cmap='gray')
        plt.title(f"3. Final Input to Model (Normalized)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"\n[Plotting Error] Failed to display image visualization. Error: {e}")
    
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