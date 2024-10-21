# Name: Jonathan Turner
# Number: 001075086
# Assignment 4
# Model Evaluation

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
from torch.optim import Adam

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Possibly add transforms to improve data?

# Checks for cuda device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loads the dataset
print('Loading dataset...')
train_dataset = datasets.ImageFolder(root='DS_IDRID/Train', transform=transforms.ToTensor())
test_dataset = datasets.ImageFolder(root='DS_IDRID/Test', transform=transforms.ToTensor())
print('Dataset loaded.')

print('Train Dataset Classes:', train_dataset.class_to_idx)
print('Test Dataset Classes:', test_dataset.class_to_idx)

# Creates the dataloaders
print('Creating dataloaders...')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
print('Dataloaders created.')


# Training
def train_model(model, train_loader, crit, optim, epochs=100):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # Sends the data to the device.
            images = images.to(device)
            labels = labels.to(device)

            optim.zero_grad()
            outputs = model(images)
            loss = crit(outputs, labels)
            loss.backward()
            optim.step()
            running_loss += loss.item()

        print(f'Epoch: {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}')


# Evaluation
def eval_model(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            # Sends data to the device
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # Fixes weird errors with gpu tensors ;-;
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
        return all_labels, all_preds


# ROC Curves
def plot_roc_curve(labels, preds, save_path, title='Receiver Operating Characteristic (ROC) Curve'):
    # False positives, true postivies, and the area under the curve.
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Plots the ROC curve.
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier (area = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")

    # Saved for use in paper.
    plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix(matrix, classes, save_path, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.savefig(save_path)
    plt.show()


# Learning rates and epoch sets
lr_set = [0.001, 0.0005, 0.0001]
epochs_set = [10, 25, 50]
accuracy_scores = []

for lr in lr_set:
    for epochs in epochs_set:
        print(f'Learning Rate: {lr}, Epochs: {epochs}')
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = model.fc.in_features

        # Converts to a binary classifier.
        model.fc = nn.Linear(num_features, 2)
        model = model.to(device)  # Sets torch to the device available (preferred GPU)

        # Setup optimizers.
        crit = nn.CrossEntropyLoss()
        optim = Adam(model.parameters(), lr=lr)

        print('Training model...')
        train_model(model, train_loader, crit, optim, epochs=epochs)
        print('Model trained.')

        labels, preds = eval_model(model, test_loader)

        # Evaluation metrics
        accuracy = np.mean(np.array(labels) == np.array(preds))
        accuracy_scores.append(f'LR: {lr}, Epochs: {epochs}, Accuracy: {accuracy:.4f}\n')

        cm = confusion_matrix(labels, preds)
        classes = ['DR', 'noDR']

        plot_roc_curve(labels, preds, f'output/roc_curve_lr_{lr}_epochs_{epochs}.png', title=f'ROC Curve (LR: {lr}, Epochs: {epochs})')
        plot_confusion_matrix(cm, classes, f'output/confusion_matrix_lr_{lr}_epochs_{epochs}.png', title=f'Confusion Matrix (LR: {lr}, Epochs: {epochs})')

# Saves the accuracy scores to a file.
with open('output/accuracy_scores.txt', 'w') as f:
    f.writelines(accuracy_scores)
    print('Accuracy scores saved to output/accuracy_scores.txt')
