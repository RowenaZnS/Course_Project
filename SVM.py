import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
from datetime import datetime
import joblib

class data_cls:
    def __init__(self, train_path, test_path):
        # Load training and test data
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)

        # Prepare training data
        self.X_train = self.train_df.drop(['labels'], axis=1)
        self.y_train = self.train_df['labels']

        # Prepare test data
        self.X_test = self.test_df.drop(['labels'], axis=1)
        self.y_test = self.test_df['labels']

        # Standardizing the features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def get_train_test_data(self):
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test

parent_directory = "result_SVM"
if not os.path.exists(parent_directory):
    os.makedirs(parent_directory)
directory_name = datetime.now().strftime('%Y%m%d_%H%M%S')
run_directory = os.path.join(parent_directory, directory_name)
os.makedirs(run_directory)

# Initialize data manager with paths to the training and test datasets
train_path = "./dataset/formated_train_simple.data"
test_path = "./dataset/formated_test_simple.data"
data_manager = data_cls(train_path, test_path)
X_train, X_test, y_train, y_test = data_manager.get_train_test_data()

# Initialize SVM classifier
model = SVC(kernel='rbf', gamma='scale')

# Train the model
model.fit(X_train, y_train)
# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix and Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(os.path.join(run_directory, 'confusion_matrix.png'))
model_path = os.path.join(run_directory, 'svm_model.joblib')
joblib.dump(model, model_path)