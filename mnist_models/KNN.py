import numpy as np
import os
import joblib # Used for saving/loading scikit-learn models
from PIL import Image 
import torch # Only needed for torchvision data loading
from torchvision import datasets, transforms

# --- SCALING AND CLASSIFICATION IMPORTS ---
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# KNN is sensitive to feature scaling, though sometimes less so than SVM, 
# standardizing data is still often best practice.
from sklearn.preprocessing import StandardScaler 

# --- Configuration ---
DOWNLOAD_ROOT = './mnist_data' 
MODEL_SAVE_PATH = 'mnist_saves/knn_scaler_model.joblib' 
CUSTOM_IMAGE_PATH = 'custom_digit.png' 

# K: The number of neighbors to check for classification
K_NEIGHBORS = 5 
# Use a subset for faster training/initial runs, as KNN training is fast, 
# but prediction/loading can be slow on the full set.
SUBSET_SIZE = 10000 

# --- 1. Data Preparation for Scikit-learn ---

def get_data_as_numpy(root, subset_size):
    """Loads MNIST data using torchvision and converts it to flattened NumPy arrays."""
    print("--- 1. Preparing Data for Scikit-learn ---")
    
    transform = transforms.Compose([transforms.ToTensor()])

    # Download and load the datasets
    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    # Convert PyTorch Datasets to flat NumPy arrays
    X_train_raw = train_dataset.data.numpy().astype(np.float32)
    y_train = train_dataset.targets.numpy()
    X_test_raw = test_dataset.data.numpy().astype(np.float32)
    y_test = test_dataset.targets.numpy()

    # Flatten the images from (N, 28, 28) to (N, 784) and scale to 0-1
    X_train_flat = X_train_raw.reshape(X_train_raw.shape[0], -1) / 255.0
    X_test_flat = X_test_raw.reshape(X_test_raw.shape[0], -1) / 255.0
    
    # Use a subset of the training data
    if subset_size < len(y_train):
        X_train_flat = X_train_flat[:subset_size]
        y_train = y_train[:subset_size]
        print(f"Using a training subset of size {subset_size} for training.")

    print(f"Training data shape: {X_train_flat.shape}")
    print(f"Testing data shape: {X_test_flat.shape}")

    return X_train_flat, y_train, X_test_flat, y_test

# --- 2. Model Training Function (KNN) ---

def train_model(X_train, y_train, k):
    """Trains the KNN model and the necessary StandardScaler."""
    print("\n--- 2. Training K-Nearest Neighbors Classifier ---")
    
    # 1. Initialize Scaler: Needed to ensure all features (pixels) contribute equally
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    print("Data scaled successfully.")
    
    # 2. Initialize and Train KNN
    # n_neighbors is the 'K' value.
    knn_model = KNeighborsClassifier(n_neighbors=k)
    
    # KNN "training" is fast; it mostly consists of storing the data points.
    print(f"Starting KNN training with K={k}...")
    knn_model.fit(X_train_scaled, y_train)
    print("KNN training complete (data stored).")
    
    # Return both the trained model and the scaler
    return knn_model, scaler

# --- 3. Evaluation Function (KNN) ---

def evaluate_model(knn_model, scaler, X_test, y_test):
    """Evaluates the KNN model on the test dataset."""
    print("\n--- 3. Evaluating Model ---")
    
    # Scale the test data using the *fitted* training scaler
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions (This prediction step can be slow for KNN)
    y_pred = knn_model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f'Test Accuracy (KNN): {100 * accuracy:.2f}%')
    return accuracy

# --- 4. Custom Prediction Function (KNN) ---

def predict_custom_image(knn_model, scaler, image_path):
    """Loads a custom image, preprocesses it, and makes a prediction using the KNN."""
    if not os.path.exists(image_path):
        print(f"\n--- ERROR: Custom image '{image_path}' not found! ---")
        print("Please ensure the image file is present to test the model.")
        return

    print(f"\n--- 4. Predicting Custom Image: {image_path} ---")
    
    try:
        # Load, convert to grayscale, and resize
        image = Image.open(image_path).convert('L') 
        image = image.resize((28, 28))

        # Thresholding
        image = image.point(lambda p: 0 if p < 128 else 255)
        
        # Convert PIL image to NumPy array and flatten
        image_np = np.array(image).astype(np.float32)
        X_custom_flat = image_np.reshape(1, -1) / 255.0 # Scale to 0-1
        
        # Inversion Check
        if X_custom_flat.mean() > 0.5:
            X_custom_flat = 1.0 - X_custom_flat
        
        # Scale the custom image using the fitted scaler (CRITICAL)
        X_custom_scaled = scaler.transform(X_custom_flat)
        
        # Make prediction
        prediction = knn_model.predict(X_custom_scaled)[0]
        
        print(f"KNN Prediction: The digit is {int(prediction)}")
        
    except Exception as e:
        print(f"Failed to predict custom image: {e}")

# --- 5. Model Saving and Loading Functions (Joblib) ---

def save_model(model, scaler, path):
    """Saves both the KNN model and the fitted scaler using joblib."""
    joblib.dump({'model': model, 'scaler': scaler}, path)
    print(f"\nModel and Scaler saved to {path}")
    
def load_model(path):
    """Loads the KNN model and the fitted scaler using joblib."""
    if os.path.exists(path):
        data = joblib.load(path)
        print(f"\nModel and Scaler successfully loaded from {path}")
        return data['model'], data['scaler']
    return None, None

# --- Main Execution ---
if __name__ == '__main__':
    # Get all data as flat NumPy arrays
    X_train, y_train, X_test, y_test = get_data_as_numpy(DOWNLOAD_ROOT, SUBSET_SIZE)

    # 1. Attempt to load saved weights
    knn_model, scaler = load_model(MODEL_SAVE_PATH)
    is_loaded = knn_model is not None

    # 2. Train only if the model wasn't loaded
    if not is_loaded:
        # Train the model and get the fitted scaler
        knn_model, scaler = train_model(X_train, y_train, K_NEIGHBORS)
        
        # Save the model after training
        save_model(knn_model, scaler, MODEL_SAVE_PATH)

    # 3. Evaluate the model
    evaluate_model(knn_model, scaler, X_test, y_test)

    # 4. Make a prediction on a custom image
    predict_custom_image(knn_model, scaler, CUSTOM_IMAGE_PATH)