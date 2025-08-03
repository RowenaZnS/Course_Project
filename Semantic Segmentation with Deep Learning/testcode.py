import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PotsdamDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.tif')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.file_list[idx])
        with rasterio.open(img_path) as src:
            img = src.read()  # (C, H, W)

        label = img[-1, :, :]  # (H, W)
        img = img[0:4, :, :]   # (4, H, W)

        img = np.transpose(img, (1, 2, 0))  # (H, W, 4)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)

        label = torch.tensor(label, dtype=torch.long)
        return img, label

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding='same')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 6, kernel_size=3, padding='same')
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.softmax(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
])

test_dataset = PotsdamDataset('Potsdam-5Fold-Splits/fold_4', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Load the best model weights
model.load_state_dict(torch.load('best_model.pth'))

# Ensure the model is in evaluation mode
model.eval()

# Initialize variables to track test loss and accuracy
test_loss = 0.0
correct_test = 0
total_test = 0

# No gradient is needed for evaluation
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        # Get the maximum probability predictions from the softmax output
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.numel()
        correct_test += (predicted == labels).sum().item()

# Calculate average loss and accuracy
test_loss /= len(test_loader)
test_accuracy = 100 * correct_test / total_test

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')