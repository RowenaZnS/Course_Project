from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

with open('wdbc.pkl', 'rb') as file:
    data = pickle.load(file)

df = pd.DataFrame(data)

feature_columns = df.drop(['id', 'malignant'], axis=1).columns.tolist()

X = df[feature_columns]
y = df['malignant']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize and train the SVM classifier
clf_svm = SVC(kernel='linear', C=1.0, random_state=42)  # Linear kernel for interpretability
clf_svm.fit(X_train, y_train)

# Evaluate the classifier
y_pred_svm = clf_svm.predict(X_test)
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))