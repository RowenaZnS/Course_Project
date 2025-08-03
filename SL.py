import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

class data_cls:
    def __init__(self, train_path):
        self.train_data = None
        self.train_path = train_path
        self._load_train_data()

    def _load_train_data(self):
        train_df = pd.read_csv(self.train_path)
        self.train_data = self._prepare_data(train_df, batch_size=64, shuffle=True)

    def _prepare_data(self, df, batch_size, shuffle):
        features = torch.tensor(df.iloc[:, :-1].values).float()
        targets = torch.tensor(df.iloc[:, -1].values).long()
        dataset = TensorDataset(features, targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_train_loader(self):
        return self.train_data

    def load_test_data(self, path, batch_size=32):
        test_df = pd.read_csv(path)
        return self._prepare_data(test_df, batch_size=batch_size, shuffle=False)

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.output(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_path = "./dataset/formated_train_simple.data"
    test_path = "./dataset/formated_test_simple.data"
    num_epochs = 1000
    hidden_size = 100
    learning_rate = 0.1
    output_size = 2  # Adjust based on the number of classes

    # Directory setup for saving results
    parent_directory = "result_SL"
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
    directory_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_directory = os.path.join(parent_directory, directory_name)
    os.makedirs(run_directory)

    data_manager = data_cls(train_path)
    train_loader = data_manager.get_train_loader()
    test_loader = data_manager.load_test_data(test_path, batch_size=64)

    sample_data, _ = next(iter(train_loader))
    input_size = sample_data.shape[1]

    model = Net(input_size, hidden_size, output_size).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    loss_array = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        all_outputs = []
        all_targets = []

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            all_outputs.append(output)
            all_targets.append(target)

        # Concatenate all outputs and targets for the epoch
        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)

        # Calculate loss for the whole epoch
        loss = criterion(all_outputs, all_targets)
        loss.backward()
        optimizer.step()

        average_loss = loss.item()
        loss_array.append(average_loss)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {average_loss:.4f}")

    # Save the model
    model_path1 = os.path.join(run_directory, 'model.pth')
    torch.save(model.state_dict(), model_path1)

    # Plot and save the loss history
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(loss_array)), loss_array)
    plt.title('Average Loss by Epoch -- SL')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.savefig(os.path.join(run_directory, 'loss_history.png'))
    plt.close()

    # Testing phase
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().tolist())
            all_targets.extend(target.cpu().tolist())

    cm = confusion_matrix(all_targets, all_preds)
    print("Classification Report:")
    print(classification_report(all_targets, all_preds))

    # Save and display the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix -- SL')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(run_directory, 'confusion_matrix.png'))
    plt.close()