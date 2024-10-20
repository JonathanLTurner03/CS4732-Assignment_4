import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
from torch.optim import Adam

# Possibly add transforms to improve data?

# Add a way to check for torch GPU and use it if available.

# Loads the dataset
print('Loading dataset...')
train_dataset = datasets.ImageFolder(root='DS_IDRID/Train', transform=transforms.ToTensor())
test_dataset = datasets.ImageFolder(root='DS_IDRID/Test', transform=transforms.ToTensor())
print('Dataset loaded.')

# Creates the dataloaders
print('Creating dataloaders...')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
print('Dataloaders created.')

## Loads the resnet50 model.
print('Loading model...')
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
print('Model loaded.')

# Converts to a binary classifier.
model.fc = nn.Linear(num_features, 2)
#model = model.to(device) # Gotta get torchGPU working for this first.

# Setup optimizers.
crit = nn.CrossEntropyLoss()
optim = Adam(model.parameters(), lr=0.001) # TODO Setup loop to test different rates for assgn


# Training
print('Training model...')
def train_model(model, train_loader, crit, optim, epochs=20):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for epoch in range(epochs):
            for images, labels in train_loader:
                # TODO Load images to torch device

                optim.zero_grad()
                outputs = model(images)

                loss = crit(outputs, labels)
                loss.backward()
                optim.step()
            print(f'Epoch: {epoch}/{epochs}, Loss: {running_loss/len(train_loader):.4f}')


train_model(model, train_loader, crit, optim)

