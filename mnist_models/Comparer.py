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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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

# 1. Paths to your TRAINED MODELS (for evaluation)
PATHS = {
    'ResNet': 'mnist_saves/resnet_model1.pth',
    'MLP': 'mnist_saves/mlp_model1.pth',
    'Logistic': 'mnist_saves/logistic_model1.pth',
    'KNN': 'knn_scaler_model.joblib',
    'SVM': 'svm_scaler_model.joblib'
}

# 2. Paths to your TRAINING LOGS (for loss plotting)
# Make sure these match the 'save_data_path' you used during training
LOSS_PATHS = {
    'ResNet': 'mnist_saves/resnet_training_data.csv',
    'MLP': 'mnist_saves/mlp_training_data.csv',
    'Logistic': 'mnist_saves/logistic_training_data.csv'
}

# --- 1. DATA LOADING ---
print("Loading Test Data...")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
test_dataset = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
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
        for _ in range(10): _ = model(input_tensor) # Warmup
        for _ in range(runs):
            start = time.time()
            _ = model(input_tensor)
            times.append(time.time() - start)
    return (sum(times) / len(times)) * 1000 # ms

def measure_inference_time_sklearn(model, scaler, input_np, runs=100):
    """Measures average time to predict ONE sample."""
    input_scaled = scaler.transform(input_np)
    times = []
    for _ in range(runs):
        start = time.time()
        _ = model.predict(input_scaled)
        times.append(time.time() - start)
    return (sum(times) / len(times)) * 1000 # ms

def calculate_metrics(y_true, y_pred):
    """Helper to calculate all 4 metrics at once."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, prec, rec, f1

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
    return calculate_metrics(y_true, y_pred)

def evaluate_sklearn(model, scaler, X, y):
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    return calculate_metrics(y, y_pred)

# --- 3. MAIN COMPARISON LOOP ---

results = []

# --- A. Evaluate PyTorch Models ---
torch_configs = [
    ('ResNet', MinimalResNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, NUM_BLOCKS)),
    ('MLP', MLP(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)),
    ('Logistic', LogisticRegression(INPUT_SIZE, NUM_CLASSES))
]

sample_tensor = torch.randn(1, 1, 28, 28).to(DEVICE)

for name, model_arch in torch_configs:
    path = PATHS.get(name)
    if not os.path.exists(path):
        print(f"Skipping {name}: Model file not found.")
        continue
        
    print(f"Evaluating {name}...")
    model = model_arch.to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    
    acc, prec, rec, f1 = evaluate_pytorch(model, test_loader)
    latency = measure_inference_time_pytorch(model, sample_tensor)
    size = get_file_size(path)
    
    results.append({
        'Model': name, 'Type': 'Deep Learning',
        'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1,
        'Latency (ms)': latency, 'Size (MB)': size
    })

# --- B. Evaluate Classical Models ---
sklearn_configs = ['KNN', 'SVM']
sample_np = np.random.rand(1, 784).astype(np.float32)

for name in sklearn_configs:
    path = PATHS.get(name)
    if not os.path.exists(path):
        print(f"Skipping {name}: Model file not found.")
        continue
        
    print(f"Evaluating {name}...")
    try:
        data = joblib.load(path)
        model = data['model']
        scaler = data['scaler']
        
        acc, prec, rec, f1 = evaluate_sklearn(model, scaler, X_test_np, y_test_np)
        latency = measure_inference_time_sklearn(model, scaler, sample_np)
        size = get_file_size(path)
        
        results.append({
            'Model': name, 'Type': 'Classical',
            'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1,
            'Latency (ms)': latency, 'Size (MB)': size
        })
    except Exception as e:
        print(f"Error evaluating {name}: {e}")

# --- 4. VISUALIZATION AND REPORT ---

if not results:
    print("No models evaluated. Check file paths.")
else:
    df = pd.DataFrame(results)
    
    # 1. Print Detailed Table
    print("\n" + "="*60)
    print("FINAL MODEL LEADERBOARD")
    print("="*60)
    df_display = df.copy()
    df_display['Accuracy'] = df_display['Accuracy'] * 100
    print(df_display.set_index('Model').to_string())

    # ==========================================
    # PLOT 1: LOSS COMPARISON (NEW ADDITION)
    # ==========================================
    plt.figure(figsize=(12, 7))
    found_any_csv = False
    
    for model_name, file_path in LOSS_PATHS.items():
        if os.path.exists(file_path):
            found_any_csv = True
            try:
                loss_df = pd.read_csv(file_path)
                # Plot Smoothed Loss
                plt.plot(loss_df['Epoch'], loss_df['Smoothed_Loss'], linewidth=2, label=f"{model_name} Loss")
                # Add dot at the end
                plt.scatter(loss_df['Epoch'].iloc[-1], loss_df['Smoothed_Loss'].iloc[-1], s=50)
            except Exception as e:
                print(f"Error reading CSV for {model_name}: {e}")
        else:
            print(f"Note: Training log not found for {model_name} at {file_path}")
            
    if found_any_csv:
        plt.title("Training Dynamics: Loss over Epochs", fontsize=14)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss (Smoothed)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("No CSV training logs found. Skipping Loss Plot.")

    # ==========================================
    # PLOT 2: METRICS DASHBOARD (2x2 Grid)
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Metrics Comparison', fontsize=16)
    axes = axes.flatten()
    
    metrics_config = [
        ('Accuracy', 'Accuracy Score', 'tab:blue'),
        ('Precision', 'Precision (Macro)', 'tab:green'),
        ('Recall', 'Recall (Macro)', 'tab:orange'),
        ('F1-Score', 'F1-Score (Macro)', 'tab:red')
    ]
    
    bar_colors = ['skyblue' if t == 'Deep Learning' else 'navajowhite' for t in df['Type']]

    for i, (col, title, _) in enumerate(metrics_config):
        ax = axes[i]
        bars = ax.bar(df['Model'], df[col], color=bar_colors, edgecolor='black', alpha=0.8)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(0.8, 1.0) # Zoom for MNIST
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # ==========================================
    # PLOT 3: TRADE-OFF (Accuracy vs Speed)
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(12, 6))

    colors = ['skyblue' if t == 'Deep Learning' else 'orange' for t in df['Type']]
    bars = ax1.bar(df['Model'], df['Accuracy'] * 100, color=colors, alpha=0.6, label='Accuracy')
    ax1.set_ylabel('Accuracy (%)', color='blue')
    ax1.set_ylim(80, 100)
    
    ax2 = ax1.twinx()
    ax2.plot(df['Model'], df['Latency (ms)'], color='red', marker='o', linewidth=2, label='Inference Time (ms)')
    ax2.set_ylabel('Latency (ms) - Lower is Better', color='red')
    
    plt.title('Trade-off Analysis: Accuracy vs. Speed')
    plt.show()
    
    # ==========================================
    # PLOT 4: EFFICIENCY (Size vs Accuracy)
    # ==========================================
    plt.figure(figsize=(8, 6))
    for i, row in df.iterrows():
        plt.scatter(row['Size (MB)'], row['Accuracy'] * 100, s=100, label=row['Model'])
        plt.text(row['Size (MB)'], row['Accuracy'] * 100, f"  {row['Model']}")
    
    plt.xlabel('Model Size (MB)')
    plt.ylabel('Accuracy (%)')
    plt.title('Efficiency: File Size vs. Performance')
    plt.grid(True, linestyle='--')
    plt.show()