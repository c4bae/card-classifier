import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm


class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)  # self.data holds an object

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property  # @property is strictly for appearance. allows me to call the function by PlayingCardDataset.classes
    def classes(self):
        return self.data.classes  # self.data.classes returns a list of the names of the subdirectories of the root


class SimpleCardClassifier(nn.Module):
    # Defines all the parts of the model
    def __init__(self, num_classes=53):
        super().__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280

        # Make a classifier
        self.classifier = nn.Linear(enet_out_size, num_classes)

    # Connects these parts and return the output
    def forward(self, x):
        x = self.features(x)

        # Flatten to change the tensor to a flat 2D block
        x = x.flatten(1)

        output = self.classifier(x)
        return output


data_dir = 'archive/train'
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # every image must have the same dimensions
    transforms.ToTensor(),  # samples must be converted into a Pytorch Tensor
])

train_folder = "archive/train"
valid_folder = "archive/valid"
test_folder = "archive/test"

train_dataset = PlayingCardDataset(train_folder, transform)
valid_dataset = PlayingCardDataset(valid_folder, transform)
test_dataset = PlayingCardDataset(test_folder, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# creates a new dictionary that swaps the keys and values
target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}

# Epoch refers to number of training loops through a dataset.
# With each epoch, model goes through training then validation
num_epoch = 5
train_losses, val_losses = [], []

# Allows the GPU to be used for training, speeds up the process a lot
device = torch.device("mps")
print(device)

model = SimpleCardClassifier(num_classes=53)
model.to(device)
# Loss function, returns a "loss" that indicates how far it deviates from the actual card
criterion = nn.CrossEntropyLoss()
# Optimizer, tells the nn, through math, how to get better predictions
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Run training loop
for epoch in tqdm(range(num_epoch), desc="Overall Training Loop"):
    # Set the model to train
    model.train()
    # Keep track of this loss as we train through an epoch
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc="Training Loop"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    train_loss = running_loss / len(train_dataset)
    train_losses.append(train_loss)

    # Set the model to evaluate
    model.eval()
    running_loss = 0.0
    with torch.no_grad():  # with torch.no_grad() means the code in this loop does not calculate gradients
        # Images refer to the tensor, and labels refer to the correct number for each image as a list
        # Because the batch size is 32, there are only 32 correct "class numbers" for the images
        for images, labels in tqdm(valid_loader, desc="Validation Loop"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            # loss.item() refers to the average loss across the whole batch
            # loss.item() * images.size(0) results in the total loss across the batch
            running_loss += loss.item() * images.size(0)
    val_loss = running_loss / len(valid_dataset) # len(valid_dataset) is all the samples across the dataset
    val_losses.append(val_loss)

    # Print epoch stats
    print(f"Epoch {epoch+1}/{num_epoch} - Train Loss: {train_loss} - Validation Loss: {val_loss}")

# Plot a graph of the losses
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.title("Loss across epochs")
# plt.show()


# Visualize predictions of trained model
def visualize_predictions(model, loader, num_images=12):
    model.eval()
    images_so_far = 0
    plt.figure(figsize=(12,8))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Outputs: returns a tensor with a grid of 32 rows (one for each image in the batch)
            # and 52 columns (one for each class). Each column (for each row) has a score, and the highest score
            # indicates the model's best guess for which card it is.

            outputs = model(inputs)

            # torch.max returns a tensor consisting of 32 integers (the batch size),
            # where each integer is the closest match (corresponding to the class)
            # to the ith image in the batch size
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size(0)):
                images_so_far += 1

                # Subplot divides plot into num_images / 3 rows, 3 columns
                ax = plt.subplot(num_images // 3, 3, images_so_far)
                ax.axis('off')

                # Prediction logic
                predicted_class = train_dataset.classes[preds[j]]
                actual_class = train_dataset.classes[labels[j]]

                color = 'green' if predicted_class == actual_class else 'red'
                ax.set_title(f'Predicted: {predicted_class}\nActual: {actual_class}', color=color)

                ax.imshow(inputs.cpu().data[j].permute(1, 2, 0))

                # Exit if requested number of images has been processed (tested)
                if images_so_far == num_images:
                    plt.show()
                    return


# Visualize the predictions made
visualize_predictions(model, train_loader)
