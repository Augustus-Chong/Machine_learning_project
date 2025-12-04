import torch
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
CUSTOM_IMAGE_PATH = 'custom_digit.png' 

# 1. Load Real MNIST Image
transform = transforms.Compose([transforms.ToTensor()])
mnist_data = datasets.MNIST(root='./mnist_data', train=False, download=True, transform=transform)
# Get a random image (e.g., index 1234)
real_mnist_tensor, label = mnist_data[12]

# 2. Load Your Custom Image (Using your exact preprocessing steps)
def load_custom(path):
    if not os.path.exists(path): return None
    img = Image.open(path).convert('L')
    
    # Simulate your current pipeline
    t = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    img_tensor = t(img)
    
    # Your inversion logic
    #if img_tensor.mean() < 0.5: img_tensor = 1 - img_tensor
        
    return img_tensor

custom_tensor = load_custom(CUSTOM_IMAGE_PATH)

# 3. Plot Side-by-Side
plt.figure(figsize=(10, 5))

# Plot MNIST
plt.subplot(1, 2, 1)
plt.imshow(real_mnist_tensor.squeeze(), cmap='gray')
plt.title(f"What Model Expects (MNIST)\nMax Val: {real_mnist_tensor.max():.2f}")
plt.axis('off')

# Plot Yours
plt.subplot(1, 2, 2)
if custom_tensor is not None:
    plt.imshow(custom_tensor.squeeze(), cmap='gray')
    plt.title(f"What You Are Feeding It\nMax Val: {custom_tensor.max():.2f}")
else:
    plt.text(0.5, 0.5, "Image Not Found", ha='center')
plt.axis('off')

plt.show()