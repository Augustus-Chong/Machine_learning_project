import torch
import torch.nn as nn
import numpy as np
import time
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from resnet import MinimalResNet, ConvResNet
from mlp import MLP
from logistic import LogisticRegression

# --- CONFIGURATION ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 784
HIDDEN_SIZE = 128
NUM_CLASSES = 10
NUM_BLOCKS = 3
DATA_ROOT = './mnist_data'

# Paths to your saved models (Update these to match your actual paths)
PATHS = {
    'ResNet': 'resnet_model1.pth',
    'MLP': 'mlp_model1.pth',
    'Logistic': 'logistic_model1.pth',
    'KNN': 'knn_scaler_model.joblib',
    'SVM': 'svm_scaler_model.joblib'
}

# --- 1. DATA LOADING ---
print("Loading Test Data...")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
test_dataset = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
# Create a large batch for PyTorch evaluation
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Prepare NumPy data for Scikit-Learn
X_test_np = test_dataset.data.numpy().astype(np.float32).reshape(-1, 784) / 255.0
y_test_np = test_dataset.targets.numpy()

# --- 2. METRIC FUNCTIONS ---

def get_file_size(path):
    """Returns file size in Megabytes (MB)."""
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0

def measure_inference_time_pytorch(model, input_tensor, runs=100):
    """Measures average time to predict ONE batch of 1 image."""
    model.eval()
    times = []
    with torch.no_grad():
        # Warmup
        for _ in range(10): _ = model(input_tensor)
        # Measure
        for _ in range(runs):
            start = time.time()
            _ = model(input_tensor)
            times.append(time.time() - start)
    return (sum(times) / len(times)) * 1000 # Convert to ms

def measure_inference_time_sklearn(model, scaler, input_np, runs=100):
    """Measures average time to predict ONE sample."""
    # Pre-scale to measure pure inference time
    input_scaled = scaler.transform(input_np)
    times = []
    for _ in range(runs):
        start = time.time()
        _ = model.predict(input_scaled)
        times.append(time.time() - start)
    return (sum(times) / len(times)) * 1000 # Convert to ms

def evaluate_pytorch(model, loader):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            outs = model(imgs)
            _, preds = torch.max(outs, 1)
            y_true.extend(lbls.numpy())
            y_pred.extend(preds.cpu().numpy())
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')

def evaluate_sklearn(model, scaler, X, y):
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    return accuracy_score(y, y_pred), f1_score(y, y_pred, average='macro')

# --- 3. MAIN COMPARISON LOOP ---

results = []

# --- A. Evaluate PyTorch Models ---
torch_configs = [
    ('ResNet', MinimalResNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, NUM_BLOCKS)),
    ('MLP', MLP(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)),
    ('Logistic', LogisticRegression(INPUT_SIZE, NUM_CLASSES))
]

sample_tensor = torch.randn(1, 1, 28, 28).to(DEVICE) # Dummy input for timing

for name, model_arch in torch_configs:
    path = PATHS.get(name)
    if not os.path.exists(path):
        print(f"Skipping {name}: File not found.")
        continue
        
    print(f"Evaluating {name}...")
    model = model_arch.to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    
    # 1. Performance
    acc, f1 = evaluate_pytorch(model, test_loader)
    
    # 2. Efficiency
    latency = measure_inference_time_pytorch(model, sample_tensor)
    size = get_file_size(path)
    
    results.append({
        'Model': name,
        'Type': 'Deep Learning',
        'Accuracy (%)': acc * 100,
        'F1-Score': f1,
        'Latency (ms)': latency,
        'Size (MB)': size
    })

# --- B. Evaluate Classical Models ---
sklearn_configs = ['KNN', 'SVM']

sample_np = np.random.rand(1, 784).astype(np.float32) # Dummy input

for name in sklearn_configs:
    path = PATHS.get(name)
    if not os.path.exists(path):
        print(f"Skipping {name}: File not found.")
        continue
        
    print(f"Evaluating {name}...")
    try:
        data = joblib.load(path)
        model = data['model']
        scaler = data['scaler']
        
        # 1. Performance
        # Note: SVM/KNN prediction on 10k images can be slow
        acc, f1 = evaluate_sklearn(model, scaler, X_test_np, y_test_np)
        
        # 2. Efficiency
        latency = measure_inference_time_sklearn(model, scaler, sample_np)
        size = get_file_size(path)
        
        results.append({
            'Model': name,
            'Type': 'Classical',
            'Accuracy (%)': acc * 100,
            'F1-Score': f1,
            'Latency (ms)': latency,
            'Size (MB)': size
        })
    except Exception as e:
        print(f"Error evaluating {name}: {e}")

# --- 4. VISUALIZATION AND REPORT ---

if not results:
    print("No models evaluated. Check file paths.")
else:
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("FINAL MODEL LEADERBOARD")
    print("="*60)
    print(df.sort_values(by='Accuracy (%)', ascending=False).to_string(index=False))
    
    # Plotting Trade-offs
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar chart for Accuracy
    colors = ['skyblue' if t == 'Deep Learning' else 'orange' for t in df['Type']]
    bars = ax1.bar(df['Model'], df['Accuracy (%)'], color=colors, alpha=0.6, label='Accuracy')
    ax1.set_ylabel('Accuracy (%)', color='blue')
    ax1.set_ylim(80, 100) # Zoom in to see differences
    
    # Line chart for Latency (Speed) on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(df['Model'], df['Latency (ms)'], color='red', marker='o', linewidth=2, label='Inference Time (ms)')
    ax2.set_ylabel('Latency (ms) - Lower is Better', color='red')
    
    plt.title('Trade-off Analysis: Accuracy vs. Speed')
    plt.show()
    
    # Scatter Plot: Size vs Accuracy
    plt.figure(figsize=(8, 6))
    for i, row in df.iterrows():
        plt.scatter(row['Size (MB)'], row['Accuracy (%)'], s=100, label=row['Model'])
        plt.text(row['Size (MB)'], row['Accuracy (%)'], f"  {row['Model']}")
    
    plt.xlabel('Model Size (MB)')
    plt.ylabel('Accuracy (%)')
    plt.title('Efficiency: File Size vs. Performance')
    plt.grid(True, linestyle='--')
    plt.show()