import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


class KMeansClassifier:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters)
        self.cluster_labels = None

    def fit(self, features, labels):
        # Train KMeans on the feature data
        self.model.fit(features)
        # Initialize an array to store the majority label for each cluster
        self.cluster_labels = np.zeros(self.n_clusters)
        for cluster_id in range(self.n_clusters):
            # Retrieve labels corresponding to points assigned to the current cluster
            cluster_members = labels[self.model.labels_ == cluster_id]
            # If there are points in the cluster, assign the majority label to the cluster
            if len(cluster_members) > 0:
                self.cluster_labels[cluster_id] = np.bincount(cluster_members).argmax()

    def predict(self, features):
        # Predict cluster assignments for new data points
        assigned_clusters = self.model.predict(features)
        predictions = []
        for cluster in assigned_clusters:
            predictions.append(self.cluster_labels[cluster])
        return np.array(predictions)

    def score(self, true_labels, predicted_labels):
        # Calculate and return the accuracy score.
        return accuracy_score(true_labels, predicted_labels)


# Load datasets from CSV files
df_beijing = pd.read_csv('Beijing_labeled.csv')
df_shenyang = pd.read_csv('Shenyang_labeled.csv')
df_guangzhou = pd.read_csv('Guangzhou_labeled.csv')
df_shanghai = pd.read_csv('Shanghai_labeled.csv')

# Combine Beijing and Shenyang data for training/validation
training_df = pd.concat([df_beijing, df_shenyang])
X_train = training_df.drop(columns=['PM_HIGH'])
y_train = training_df['PM_HIGH']

# Separate test data for Guangzhou
X_test_guangzhou = df_guangzhou.drop(columns=['PM_HIGH'])
y_test_guangzhou = df_guangzhou['PM_HIGH']

# Separate test data for Shanghai
X_test_shanghai = df_shanghai.drop(columns=['PM_HIGH'])
y_test_shanghai = df_shanghai['PM_HIGH']

# Initialize KMeansClassifier
kmeans = KMeansClassifier(n_clusters=2)

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store accuracies for each fold
train_accuracies = []
val_accuracies = []

# Track the best model based on validation accuracy
best_val_accuracy = 0
best_model = None

# Perform KFold cross-validation
for fold, (train_index, val_index) in enumerate(kf.split(X_train), 1):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    # Train the model on the current fold
    kmeans.fit(X_train_fold, y_train_fold)

    # Calculate training accuracy for the current fold
    train_predictions = kmeans.predict(X_train_fold)
    train_accuracy = kmeans.score(y_train_fold, train_predictions)
    train_accuracies.append(train_accuracy)

    # Calculate validation accuracy for the current fold
    val_predictions = kmeans.predict(X_val_fold)
    val_accuracy = kmeans.score(y_val_fold, val_predictions)
    val_accuracies.append(val_accuracy)

    # Track the best model based on validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model = kmeans  # Save the best model

    print(f"Fold {fold}:")
    print(f"  Training Accuracy: {train_accuracy:.4f}")
    print(f"  Validation Accuracy: {val_accuracy:.4f}")
    print("-" * 30)

# Print average validation accuracy
print(f"Average Training Accuracy: {np.mean(train_accuracies):.4f}")
print(f"Average Validation Accuracy: {np.mean(val_accuracies):.4f}")

# Use the best model from KFold to evaluate on the test set

guangzhou_predictions = best_model.predict(X_test_guangzhou)
guangzhou_accuracy = best_model.score(y_test_guangzhou, guangzhou_predictions)

shanghai_predictions = best_model.predict(X_test_shanghai)
shanghai_accuracy = best_model.score(y_test_shanghai, shanghai_predictions)

print(f"Test Accuracy (Guangzhou): {guangzhou_accuracy:.4f}")
print(f"Test Accuracy (Shanghai): {shanghai_accuracy:.4f}")
