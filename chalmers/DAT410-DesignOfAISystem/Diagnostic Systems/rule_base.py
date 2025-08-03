import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report

with open("wdbc.pkl", "rb") as f:
    data = pickle.load(f)

features = data.drop(["id", "malignant"], axis=1)

labels = data["malignant"]

rule_mapping = {
    "cell_size": [
        "radius_2",
        "area_2",
        "perimeter_2"
    ],
    "cell_shape": [
        "concavity_1",
        "concave points_1",
        "compactness_1",
        "fractal dimension_1"
    ],
    "cell_texture": [
        "texture_1"
    ],
    "cell_homogeneity": [
        "smoothness_2",
        "symmetry_2"
    ]
}


benign_samples = features[labels == 0]
thresholds = {}

for category, feats in rule_mapping.items():
    # category = cell size, feats= ['radius_2', 'area_2', 'perimeter_2']
    for feat in feats:
        thresholds[feat] = benign_samples[feat].quantile(0.95)

def rule_based_classifier(row):
    size_abnormal = any(row[feat] > thresholds[feat] for feat in rule_mapping["cell_size"])
    shape_abnormal = any(row[feat] > thresholds[feat] for feat in rule_mapping["cell_shape"])
    texture_abnormal = any(row[feat] > thresholds[feat] for feat in rule_mapping["cell_texture"])
    homogeneity_abnormal = any(row[feat] > thresholds[feat] for feat in rule_mapping["cell_homogeneity"])

    if size_abnormal or shape_abnormal or texture_abnormal or homogeneity_abnormal:
        return 1
    else:
        return 0

predictions = features.apply(rule_based_classifier, axis=1)

print("Accuracy:", accuracy_score(labels, predictions))

print("\nEvluation Report:")
print(classification_report(labels, predictions))