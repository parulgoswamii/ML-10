
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Step 1: Create Models Folder
os.makedirs("models", exist_ok=True)

# Step 2: Define Symptoms
symptoms = [
    "Fever",
    "Cough",
    "Headache",
    "Fatigue",
    "Nausea",
    "Joint_Pain",
    "Rash",
    "Sore_Throat",
    "Breathing_Difficulty",
    "Body_Pain"
]

# Step 3: Disease Profiles
disease_profiles = {
    "Common Cold": [0.2, 0.9, 0.5, 0.5, 0.1, 0.1, 0.1, 0.9, 0.1, 0.3],
    "Flu": [0.9, 0.7, 0.7, 0.9, 0.3, 0.6, 0.1, 0.4, 0.2, 0.9],
    "Migraine": [0.1, 0.1, 0.95, 0.6, 0.8, 0.1, 0.0, 0.1, 0.1, 0.2],
    "Allergy": [0.1, 0.5, 0.2, 0.3, 0.1, 0.1, 0.8, 0.2, 0.2, 0.1],
    "Gastroenteritis": [0.5, 0.1, 0.3, 0.7, 0.95, 0.1, 0.1, 0.1, 0.1, 0.4],
    "Arthritis": [0.1, 0.1, 0.2, 0.5, 0.1, 0.95, 0.1, 0.1, 0.1, 0.6]
}

# Step 4: Generate Sample Dataset
np.random.seed(42)

data = []

for disease, probabilities in disease_profiles.items():

    for i in range(200):

        patient = {
            symptom: np.random.binomial(1, probability)
            for symptom, probability in zip(symptoms, probabilities)
        }

        patient["Age"] = np.random.randint(18, 80)
        patient["Disease"] = disease

        data.append(patient)

df = pd.DataFrame(data)

# Step 5: Save Dataset
df.to_csv("disease_dataset.csv", index=False)

print("\nDisease Dataset Created Successfully!")
print(df.head())

# Step 6: Dataset Information
print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDisease Distribution:")
print(df["Disease"].value_counts())

# Step 7: Separate Features and Target
X = df.drop("Disease", axis=1)
y = df["Disease"]

# Step 8: Data Preprocessing
preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Step 9: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 10: Machine Learning Algorithms
algorithms = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )
}

best_model = None
best_accuracy = 0
best_name = ""

# Step 11: Train and Compare Models
for name, algorithm in algorithms.items():

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", algorithm)
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n================================")
    print("Algorithm:", name)
    print("Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_name = name

# Step 12: Save Best Model
joblib.dump(best_model, "models/disease_model.joblib")

print("\n================================")
print("Best Algorithm:", best_name)
print("Best Accuracy:", round(best_accuracy * 100, 2), "%")
print("Model Saved Successfully!")
print("Disease Prediction Training Completed!")
