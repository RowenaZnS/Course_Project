import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

with open('wdbc.pkl', 'rb') as file:
    data = pickle.load(file)

df = pd.DataFrame(data)

feature_columns = df.drop(['id', 'malignant'], axis=1).columns.tolist()

X = df[feature_columns]
y = df['malignant']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))