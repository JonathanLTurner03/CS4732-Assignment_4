import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# Possibly add transforms to improve data?

# Loads the dataset
train_dataset = datasets.ImageFolder(root='DS_IDRID/Train', transform=transforms.ToTensor())
test_dataset = datasets.ImageFolder(root='DS_IDRID/Test', transform=transforms.ToTensor())

# Creates the dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

## Loads the resnet50 model.
model = models.resnet50(pretrained=True)