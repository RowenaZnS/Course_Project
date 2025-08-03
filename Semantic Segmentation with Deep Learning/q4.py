import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Define a dummy dataset class with geographic coordinates
import torch
from torch.utils.data import Dataset
import numpy as np

class GeoDataset(Dataset):
    def __init__(self, n_samples):
        np.random.seed(42)
        self.n_samples = n_samples
        # Simulate latitude/longitude
        self.latitudes = np.random.uniform(-90, 90, n_samples)
        self.longitudes = np.random.uniform(-180, 180, n_samples)
        # Simulate input features (5 channels, 224x224) and labels (224x224)
        self.features = [torch.randn(5, 224, 224) for _ in range(n_samples)]
        self.labels = [torch.randint(0, 6, (224, 224)) for _ in range(n_samples)]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Handle batched indices
        if isinstance(idx, (list, np.ndarray, torch.Tensor)):
            batch = []
            for i in idx:
                batch.append(self.__getitem__(i))  # Recursively fetch each item
            return batch

        # Single index case
        feature = self.features[idx]
        label = self.labels[idx]
        return {
            "features": feature,
            "label": label,
            "latitude": self.latitudes[idx],
            "longitude": self.longitudes[idx],
        }
# Initialize the dataset
dataset = GeoDataset(1000)  # Example: 1000 samples

# Scatter plot to visualize latitude/longitude
latitudes = [dataset[i]["latitude"] for i in range(len(dataset))]
longitudes = [dataset[i]["longitude"] for i in range(len(dataset))]
plt.scatter(longitudes, latitudes, alpha=0.5)
plt.title("Geographic Distribution of the Dataset")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Split the dataset into 5 geographic folds using KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = list(kf.split(range(len(dataset))))

# Train/Validation/Test split
train_indices, test_indices = folds[0]  # Fold 1 as the test set
val_indices = folds[1]  # Fold 2 as the validation set
train_indices = np.concatenate([folds[i][0] for i in range(2, 5)])  # Remaining folds as training set

# Create data loaders for training, validation, and testing
batch_size = 8
from torchvision.transforms import functional as F
def custom_collate_fn(batch):
    # Extract features, labels, latitudes, and longitudes
    features = [item["features"] for item in batch]
    labels = [item["label"] for item in batch]
    latitudes = [item["latitude"] for item in batch]
    longitudes = [item["longitude"] for item in batch]

    # Find the maximum height and width in the batch
    max_height = max(f.shape[1] for f in features)
    max_width = max(f.shape[2] for f in features)

    # Pad all features and labels to the same size
    padded_features = [F.pad(f, (0, max_width - f.shape[2], 0, max_height - f.shape[1])) for f in features]
    padded_labels = [F.pad(l, (0, max_width - l.shape[1], 0, max_height - l.shape[0])) for l in labels]

    # Stack features and labels into tensors
    features = torch.stack(padded_features)
    labels = torch.stack(padded_labels)

    # Convert latitudes and longitudes to tensors
    latitudes = torch.tensor(latitudes)
    longitudes = torch.tensor(longitudes)

    return {"features": features, "label": labels, "latitude": latitudes, "longitude": longitudes}
train_loader = DataLoader(Subset(dataset, train_indices), batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(Subset(dataset, val_indices), batch_size=8, shuffle=False, collate_fn=custom_collate_fn)
test_loader = DataLoader(Subset(dataset, test_indices), batch_size=8, shuffle=False, collate_fn=custom_collate_fn)

# Define the EncoderDecoderCNN model
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
        self.t_conv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
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

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model, loss function, and optimizer
model = EncoderDecoderCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, val_loader, epochs):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                labels = batch["label"].to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Train the model
train_model(model, train_loader, val_loader, epochs=20)

# Evaluate the model on the test set
def evaluate_model(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

# Evaluate on the test set
evaluate_model(model, test_loader)