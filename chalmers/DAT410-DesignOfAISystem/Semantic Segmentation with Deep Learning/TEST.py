import os
import random
from sklearn.model_selection import KFold
from shutil import copyfile

# Path to the directory containing GeoTIFF files
PathToGeoTiffFile = "Potsdam-GeoTif"

# Output folder for 5-fold splits
output_folder = "Potsdam-5Fold-Splits"
os.makedirs(output_folder, exist_ok=True)

# Step 1: Read all GeoTIFF file paths
all_files = [os.path.join(PathToGeoTiffFile, f) for f in os.listdir(PathToGeoTiffFile) if f.endswith(".tif")]

# Ensure there are enough files to sample from
if len(all_files) < 10000:
    print(f"Warning: Only {len(all_files)} files found. Adjusting sample size.")
    sampled_files = all_files  # Use all files if less than 10000
else:
    # Step 2: Randomly sample 10,000 images
    random.seed(42)  # Ensure reproducibility
    sampled_files = random.sample(all_files, 10000)

# Step 3: Split into 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (_, fold_idx) in enumerate(kf.split(sampled_files)):
    fold_folder = os.path.join(output_folder, f"fold_{fold}")
    os.makedirs(fold_folder, exist_ok=True)

    # Copy files for this fold
    for idx in fold_idx:
        src = sampled_files[idx]
        dst = os.path.join(fold_folder, os.path.basename(src))
        try:
            copyfile(src, dst)
        except Exception as e:  # Catch any exceptions during copy
            print(f"Error copying file {src} to {dst}: {e}")

    print(f"Fold {fold} prepared: {len(fold_idx)} files")

print(f"5-Fold splits saved in: {output_folder}")



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
if(device == torch.device("cuda")):
    print("Using GPU")
else :
    print("Using CPU")
class PotsdamDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.file_list = []
        for f in os.listdir(folder_path):
            if f.endswith('.tif'):
                file_path = os.path.join(folder_path, f)
                if os.path.getsize(file_path) > 0:
                    self.file_list.append(f)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.file_list[idx])
        with rasterio.open(img_path) as src:
            img = src.read()

        label = img[-1, :, :]
        img = img[0:4, :, :]
        img = np.transpose(img, (1, 2, 0))

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        label = torch.tensor(label, dtype=torch.long)
        return img, label

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.conv4 = nn.Conv2d(128, 6, kernel_size=1, padding='same')
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.upsample(x)
        x = self.softmax(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
])

train_folders = ['Potsdam-5Fold-Splits/fold_0', 'Potsdam-5Fold-Splits/fold_1', 'Potsdam-5Fold-Splits/fold_2']
train_datasets = [PotsdamDataset(folder, transform=transform) for folder in train_folders]
train_dataset = ConcatDataset(train_datasets)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataset = PotsdamDataset('Potsdam-5Fold-Splits/fold_3', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_dataset = PotsdamDataset('Potsdam-5Fold-Splits/fold_4', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    best_val_loss = float('inf')
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

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)

model.load_state_dict(torch.load('best_model.pth'))
model.eval()

test_loss = 0.0
correct_test = 0
total_test = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

test_loss /= len(test_loader)
test_accuracy = 100 * correct_test / total_test
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')