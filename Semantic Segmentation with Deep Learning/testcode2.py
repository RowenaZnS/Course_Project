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
model = EncoderDecoderCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
def test_model(model, test_loader, criterion):
    model.eval()  # Ensure model is in eval mode for testing
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # No gradients needed for testing
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Shape: (batch_size, 6, 224, 224)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predictions = torch.max(outputs, 1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.numel()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct_predictions / total_predictions
    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

# Load the best model weights
model.load_state_dict(torch.load('best_model3.pth'))
test_dataset = PotsdamDataset('Potsdam-5Fold-Splits/fold_4', transform=transform)

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
# Evaluate the model on the test set
test_model(model, test_loader, criterion)