import pandas as pd
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

with open('wdbc.pkl', 'rb') as file:
    data = pickle.load(file)
df = pd.DataFrame(data)
feature_columns = df.drop(['id', 'malignant'], axis=1).columns.tolist()
X = df[feature_columns].values
y = df['malignant'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


class ClassifyNet(nn.Module):
    def __init__(self):
        super(ClassifyNet, self).__init__()
        self.layer1 = nn.Linear(X_train.shape[1], 30)
        self.layer2 = nn.Linear(30, 15)
        self.output_layer = nn.Linear(15, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output_layer(x))
        return x


model = ClassifyNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 200
losses = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs.squeeze(), y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.title('Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

model.eval()
with torch.no_grad():
    outputs = model(X_test)
    y_pred = (outputs.squeeze() >= 0.5).float()
    print("Neural Network Classification Report (PyTorch):\n", classification_report(y_test.numpy(), y_pred.numpy()))