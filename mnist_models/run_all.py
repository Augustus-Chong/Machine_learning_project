import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import joblib
from PIL import Image, ImageOps 
from torchvision import transforms

from resnet import MinimalResNet
from mlp import MLP
from logistic import LogisticRegression
from Model_Base import load_model, predict

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
    Loads custom image and applies necessary normalization and scaling
    to generate the tensor (for ResNet) and the numpy array (for KNN/SVM).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Custom image '{image_path}' not found.")

    image = Image.open(image_path).convert('L') 
    
    # 1. Define simplified transformation pipeline (PIL to Tensor, 0-1 scaling)
    simple_transform = transforms.Compose([
        transforms.Resize((28, 28)), 
        transforms.ToTensor(),       
    ])
    
    image_tensor = simple_transform(image).unsqueeze(0)

    # 2. Critical Inversion Check
    if image_tensor.mean() < 0.5:
        image_tensor = 1 - image_tensor 

    # 3. Apply Standard MNIST Normalization (PyTorch Tensor format)
    mean_tensor = torch.tensor([0.1307]).view(1, 1, 1, 1)
    std_tensor = torch.tensor([0.3081]).view(1, 1, 1, 1)
    normalized_tensor = (image_tensor - mean_tensor) / std_tensor
    
    return normalized_tensor.to(device)

if __name__ == '__main__':
    try:
        # 1. Preprocess the custom image once
        models_input = preprocess_custom_image(CUSTOM_IMAGE_PATH)
        
        # 2. Load Models
        Rmodel = MinimalResNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, NUM_RESIDUAL_BLOCKS).to(device)
        resnet_model = load_model(Rmodel, MODEL_PATH_RESNET, device)

        Mmodel = MLP(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
        mlp_model = load_model(Mmodel, MODEL_PATH_MLP, device)

        Lmodel = LogisticRegression(INPUT_SIZE, NUM_CLASSES).to(device)
        logistic_model = load_model(Lmodel, MODEL_PATH_LOGISTIC, device)

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