import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
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
        img = img[:-1, :, :]   # (C, H, W) where C should be 5 here, including RGB, IR, Elevation

        # Convert to (H, W, C) for compatibility with ToTensor()
        img = np.transpose(img, (1, 2, 0))  # (H, W, C)

        if self.transform:
            img = self.transform(img)
        else:
            # Ensure output is a PyTorch tensor even without transformations
            img = torch.from_numpy(img).float().permute(2, 0, 1)  # Convert numpy array to tensor

        label = torch.from_numpy(label).long()  # Convert numpy array to tensor
        return img, label


import torch
import torch.nn as nn


class EncoderDecoderCNN(nn.Module):
    def __init__(self):
        super(EncoderDecoderCNN, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu4 = nn.ReLU()
        self.t_conv2 = nn.ConvTranspose2d(128, 32, kernel_size=3,stride=2, padding=1, output_padding=1)
        self.relu5 = nn.ReLU()
        self.final_conv = nn.Conv2d(32, 6, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=1)
        self.t_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
    def forward(self, x):
        x1 = self.relu1(self.conv1(x))
        x2 = self.pool1(x1)
        x3 = self.relu2(self.conv2(x2))
        x4 = self.pool2(x3)
        x5 = self.relu3(self.conv3(x4))


        x6 = self.relu4(self.t_conv1(x5))


        x7 = torch.cat((x3, x6), 1)

        x8 = self.relu4(self.t_conv2(x7))

        x9 = torch.cat((x8, x1), 1)



        x10 = self.relu4(self.t_conv3(x9))

        output = self.softmax(self.final_conv(x10))
        return output

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.485, 0.456], std=[0.229, 0.224, 0.225, 0.229, 0.224]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
])

train_folders = [
    'Potsdam-5Fold-Splits/fold_0',
    'Potsdam-5Fold-Splits/fold_1',
    'Potsdam-5Fold-Splits/fold_2'
]
train_datasets = [PotsdamDataset(folder, transform=transform) for folder in train_folders]
train_dataset = ConcatDataset(train_datasets)

# 创建 DataLoader，设置 shuffle=True
train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True)
val_dataset = PotsdamDataset('Potsdam-5Fold-Splits/fold_3', transform=transform)
test_dataset = PotsdamDataset('Potsdam-5Fold-Splits/fold_4', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = EncoderDecoderCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.numel()
            correct_train += (predicted == labels).sum().item()
            print(labels.numel()==labels.size(1)*labels.size(2)*8)


        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.numel()
                correct_val += (predicted == labels).sum().item()


        val_loss /= len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model3.pth')

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

    return train_losses, val_losses, train_accuracies, val_accuracies

train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


