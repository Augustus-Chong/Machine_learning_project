import numpy as np
import os
import joblib # Used for saving/loading scikit-learn models
from PIL import Image 
import torch # Only needed for torchvision data loading
from torchvision import datasets, transforms

# --- SCALING AND CLASSIFICATION IMPORTS ---
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# --- Configuration ---
DOWNLOAD_ROOT = './mnist_data' 
MODEL_SAVE_PATH = 'mnist_saves/svm_scaler_model.joblib' 
CUSTOM_IMAGE_PATH = 'custom_digit.png' 

# C parameter controls regularization (lower C = more regularization)
# SVM training is significantly slower than PyTorch, so we use a small subset of the data
# for demonstration, but keep the full MNIST data loaded for reference.
SUBSET_SIZE = 10000 
SVM_C = 5.0 # A common default for RBF kernel

# --- 1. Data Preparation for Scikit-learn ---

def get_data_as_numpy(root, subset_size):
    """Loads MNIST data using torchvision and converts it to flattened NumPy arrays."""
    print("--- 1. Preparing Data for Scikit-learn ---")
    
    # We use ToTensor but not Normalize, as StandardScaler will handle normalization
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
    
    # Use a subset of the training data to make the initial training faster
    if subset_size < len(y_train):
        X_train_flat = X_train_flat[:subset_size]
        y_train = y_train[:subset_size]
        print(f"Warning: Using a training subset of size {subset_size} for faster demonstration.")

    print(f"Training data shape: {X_train_flat.shape}")
    print(f"Testing data shape: {X_test_flat.shape}")

    # Return raw features and labels
    return X_train_flat, y_train, X_test_flat, y_test

# --- 2. Model Training Function (SVM) ---

def train_model(X_train, y_train, C):
    """Trains the SVM model and the necessary StandardScaler."""
    print("\n--- 2. Training Support Vector Machine ---")
    
    # 1. Initialize Scaler: SVM is highly sensitive to the scale of data
    scaler = StandardScaler()
    
    # Fit the scaler ONLY on the training data and transform it
    X_train_scaled = scaler.fit_transform(X_train)
    print("Data scaled successfully.")
    
    # 2. Initialize and Train SVM
    # The 'rbf' kernel is a non-linear choice often yielding high accuracy
    svm_model = SVC(kernel='rbf', C=C, verbose=True)
    
    # Training the SVM (This step is computationally intensive)
    print("Starting SVM training (may take a few minutes)...")
    svm_model.fit(X_train_scaled, y_train)
    print("SVM training complete.")
    
    # Return both the trained model and the scaler (both needed for prediction)
    return svm_model, scaler

# --- 3. Evaluation Function (SVM) ---

def evaluate_model(svm_model, scaler, X_test, y_test):
    """Evaluates the SVM model on the test dataset."""
    print("\n--- 3. Evaluating Model ---")
    
    # Scale the test data using the *fitted* training scaler
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = svm_model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f'Test Accuracy (SVM): {100 * accuracy:.2f}%')
    return accuracy

# --- 4. Custom Prediction Function (SVM) ---

def predict_custom_image(svm_model, scaler, image_path):
    """Loads a custom image, preprocesses it, and makes a prediction using the SVM."""
    if not os.path.exists(image_path):
        print(f"\n--- ERROR: Custom image '{image_path}' not found! ---")
        print("Please ensure the image file is present to test the model.")
        return

    print(f"\n--- 4. Predicting Custom Image: {image_path} ---")
    
    try:
        # Load, convert to grayscale, and resize
        image = Image.open(image_path).convert('L') 
        image = image.resize((28, 28))

        # Thresholding: Pure black and white image for cleaner prediction
        image = image.point(lambda p: 0 if p < 128 else 255)
        
        # Convert PIL image to NumPy array and flatten
        image_np = np.array(image).astype(np.float32)
        X_custom_flat = image_np.reshape(1, -1) / 255.0 # Scale to 0-1
        
        # Inversion Check: If the digit is dark on a light background, invert it 
        # to match the MNIST style (light digit on dark background)
        if X_custom_flat.mean() > 0.5:
            X_custom_flat = 1.0 - X_custom_flat
        
        # Scale the custom image using the fitted scaler
        # This is a CRITICAL step for SVM
        X_custom_scaled = scaler.transform(X_custom_flat)
        
        # Make prediction
        prediction = svm_model.predict(X_custom_scaled)[0]
        
        print(f"SVM Prediction: The digit is {int(prediction)}")
        
    except Exception as e:
        print(f"Failed to predict custom image: {e}")

# --- 5. Model Saving and Loading Functions (Joblib) ---

def save_model(model, scaler, path):
    """Saves both the SVM model and the fitted scaler using joblib."""
    joblib.dump({'model': model, 'scaler': scaler}, path)
    print(f"\nModel and Scaler saved to {path}")
    
def load_model(path):
    """Loads the SVM model and the fitted scaler using joblib."""
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
    svm_model, scaler = load_model(MODEL_SAVE_PATH)
    is_loaded = svm_model is not None

    # 2. Train only if the model wasn't loaded
    if not is_loaded:
        # Train the model and get the fitted scaler
        svm_model, scaler = train_model(X_train, y_train, SVM_C)
        
        # Save the model after training
        save_model(svm_model, scaler, MODEL_SAVE_PATH)

    # 3. Evaluate the model
    evaluate_model(svm_model, scaler, X_test, y_test)

    # 4. Make a prediction on a custom image
    predict_custom_image(svm_model, scaler, CUSTOM_IMAGE_PATH)