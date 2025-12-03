import torch
import torch.nn as nn
import torch.optim as optim
from Model_Base import get_data_loaders, train_model, evaluate_model, save_model, load_model, predict_custom_image, plot_loss

import time

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 100
DOWNLOAD_ROOT = './mnist_data'
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10
CUSTOM_IMAGE_PATH = 'custom_digit.png' 
MODEL_SAVE_PATH = 'mnist_saves/logistic_model.pth'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1) #flatten image into vector
        out = self.linear(x)
        return out


if __name__ == '__main__':
    #Get dataloaders
    train_loader, test_loader = get_data_loaders(DOWNLOAD_ROOT, BATCH_SIZE)
    #Initialize model
    model = LogisticRegression(INPUT_SIZE, NUM_CLASSES).to(device)

    #load saved weights
    is_loaded = load_model(model, MODEL_SAVE_PATH, device)

    if not is_loaded:
        #Define loss function
        criterion  = nn.CrossEntropyLoss()
        #Define optimizer
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
       
        start_time = time.time()
        #Train the model
        loss_data = train_model(model, train_loader, criterion, optimizer, EPOCHS, device)

        end_time = time.time()
        training_duration = end_time - start_time
        
        print(f"\nTraining Complete in {training_duration:.2f} seconds.")

        save_model(model, MODEL_SAVE_PATH)
        #Evaluate model
        evaluate_model(model, test_loader, device)

        #visualing loss
        plot_loss(loss_data, window=20)
    else:
        print("\nSkipping training as weights were loaded.")
        evaluate_model(model, test_loader, device)

    predict_custom_image(model, CUSTOM_IMAGE_PATH, device)
