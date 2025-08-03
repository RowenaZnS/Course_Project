import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

# 数据预处理
data = pd.read_csv('student-por.csv')

# 计算平均成绩
data['average_grade'] = data[['G1', 'G2', 'G3']].mean(axis=1)

# 数据分箱
quantiles = data['average_grade'].quantile([0.25, 0.5, 0.75, 1.0]).values
bins = [0] + quantiles.tolist()
labels = [0, 1, 2, 3]  # 使用数字标签代替文本标签
data['GradeClass'] = pd.cut(data['average_grade'], bins=bins, labels=labels, include_lowest=True)

# 保存连续值标签和分类标签
y_numerical = data['average_grade']
y_class = data['GradeClass'].astype(int)  # 转换为整数类型

# 删除原始分数列
data = data.drop(columns=['G1', 'G2', 'G3'])

# 二值化处理
binary_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus']
for col in binary_columns:
    unique_values = data[col].unique()
    data[col] = data[col].map({unique_values[0]: 0, unique_values[1]: 1})

# 多分类列独热编码
multi_category_columns = ['Mjob', 'Fjob', 'reason', 'guardian']
data = pd.get_dummies(data, columns=multi_category_columns, drop_first=True)

# 是/否列二值化
binary_yes_no_columns = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
for col in binary_yes_no_columns:
    data[col] = data[col].map({'yes': 1, 'no': 0})

# 特征和标签分离
X = data.drop(columns=['GradeClass'])
y = y_class

# 标准化特征
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# PCA 降维
pca = PCA(n_components=30)  # 降到 30 个主成分（根据数据调整）
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)  # 分类问题需要 long 类型
X_test_tensor = torch.tensor(X_test_pca, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# 定义神经网络模型
class ImprovedNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(ImprovedNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# 初始化模型、损失函数和优化器
input_size = X_train_pca.shape[1]
output_size = len(labels)  # 分类问题的输出大小为类别数
model = ImprovedNeuralNetwork(input_size, output_size)
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = StepLR(optimizer, step_size=5000, gamma=0.1)

# 训练循环
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()

    # 前向传播
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 更新学习率
    scheduler.step()

    # 打印每个 epoch 的损失
    if (epoch + 1) % 100 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Learning Rate: {current_lr:.6f}")

# 模型评估
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 模型评估
model.eval()
with torch.no_grad():
    y_pred_probs = model(X_test_tensor)  # 输出类别的概率分布
    y_pred = torch.argmax(y_pred_probs, dim=1).numpy()  # 获取预测类别

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)

# 打印混淆矩阵
print("Confusion Matrix:")
print(conf_matrix)

# 计算 Precision, Recall, F1-Score, Accuracy
precision = precision_score(y_test, y_pred, labels=labels, average=None)  # 每个类别的 Precision
recall = recall_score(y_test, y_pred, labels=labels, average=None)        # 每个类别的 Recall
f1 = f1_score(y_test, y_pred, labels=labels, average=None)                # 每个类别的 F1-Score
accuracy = accuracy_score(y_test, y_pred)                                # 整体 Accuracy

# 打印每个类别的指标
for i, label in enumerate(['Fail', 'Satisfactory', 'Good', 'Excellent']):
    print(f"\nClass {label}:")
    print(f"  Precision: {precision[i]:.4f}")
    print(f"  Recall: {recall[i]:.4f}")
    print(f"  F1-Score: {f1[i]:.4f}")

# 打印整体指标
macro_precision = precision_score(y_test, y_pred, labels=labels, average='macro')
macro_recall = recall_score(y_test, y_pred, labels=labels, average='macro')
macro_f1 = f1_score(y_test, y_pred, labels=labels, average='macro')

weighted_precision = precision_score(y_test, y_pred, labels=labels, average='weighted')
weighted_recall = recall_score(y_test, y_pred, labels=labels, average='weighted')
weighted_f1 = f1_score(y_test, y_pred, labels=labels, average='weighted')

print("\nOverall Metrics:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Macro Precision: {macro_precision:.4f}")
print(f"  Macro Recall: {macro_recall:.4f}")
print(f"  Macro F1-Score: {macro_f1:.4f}")
print(f"  Weighted Precision: {weighted_precision:.4f}")
print(f"  Weighted Recall: {weighted_recall:.4f}")
print(f"  Weighted F1-Score: {weighted_f1:.4f}")

# 绘制混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Fail', 'Satisfactory', 'Good', 'Excellent'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: Neural Network Classification")
plt.show()


