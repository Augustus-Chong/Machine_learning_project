import os
import torch
import torch.nn as nn
import torchvision.transforms
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class UNetClassifier:
    def __init__(self, model, data_dir, batch_size=8, input_size=(572, 572), num_epochs=10, learning_rate=0.001):
        self.model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.train_loader, self.val_loader = self.setup_data_loaders()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def setup_data_loaders(self):
        transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Load datasets using ImageFolder
        dataset = datasets.ImageFolder(root=self.data_dir, transform=transform)
        train_size = int(0.8 * len(dataset))  # 80% for training
        val_size = len(dataset) - train_size   # 20% for validation

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader

    def train(self):
        self.model.train()  # Set the model to training mode
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for images, labels in self.train_loader:
                images = images.to(next(self.model.parameters()).device)
                labels = labels.to(next(self.model.parameters()).device)

            # Zero the parameter gradients
                self.optimizer.zero_grad()

            # Forward pass
                outputs = self.model(images)

            # Average pooling to get outputs in shape (N, C)
                outputs = torch.mean(outputs, dim=[2, 3])  # Remove spatial dimensions

            # Calculate loss
                loss = self.criterion(outputs, labels)

            # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {running_loss / len(self.train_loader):.4f}')
    
        torch.save(self.model.state_dict(), 'unet_model.pth')

    def print_classification_results(self):
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for images, labels in self.val_loader:
                device = next(self.model.parameters()).device  # Get the device from the model
                images = images.to(device)  # Move images to the appropriate device
                labels = labels.to(device)  # Move labels to the appropriate device
                outputs = self.model(images)  # Get model outputs
                probabilities = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities

            # Get the predicted class
                _, predicted = torch.max(probabilities, 1)

            # Print results
                for i in range(len(images)):
                    print(f"Predicted: {self.val_loader.dataset.classes[predicted[i]]}, "
                        f"Actual: {self.val_loader.dataset.classes[labels[i]]}")

class Block(nn.Module): #inherit from Module
    def __init__(self, in_ch, out_ch):
        super().__init__() #initializer of module
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3) #output channels = how many kernels needed, kernel dims = 64 of 3x3x[3](input channels)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3)
    
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)): #channel sizes
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) #creates blocks accoding to how many different channels pairs defined in chs
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=3, retain_dim=False, out_sz=(572,572)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
        self.out_sz = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, size = self.out_sz)
        return out

def load_model(model_class, model_path):
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model

'''
# Load the model
loaded_model = load_model(UNet, 'unet_model.pth')

# Create a new classifier instance if needed
classifier = UNetClassifier(model=loaded_model, data_dir='dataset')

# Print classification results
classifier.print_classification_results()
'''
# Instantiate your UNet model
unet = UNet()  

# Create the classifier
classifier = UNetClassifier(model=unet, data_dir='dataset')

# Train the model
classifier.train()

# Print classification results
classifier.print_classification_results()