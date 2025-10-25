import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
import joblib
import os


cols = [
    "loan_id", "dependents", "education", "self_employed", "imcome_annum",
    "loan_amount", "loan_term", "cibil_score", "residential_assets_value",
    "commercial_assets_value", "luxury_assets_value", "bank_asset_value", "status"
]

df = pd.read_csv("loan_approval_dataset_processed.csv", names=cols, header=0)


if "loan_id" in df.columns:
    df = df.drop("loan_id", axis=1)
    print("ℹ️  'loan_id' column dropped automatically.")


label_enc = LabelEncoder()
if "education" in df.columns:
    df["education"] = label_enc.fit_transform(df["education"])
if "self_employed" in df.columns:
    df["self_employed"] = label_enc.fit_transform(df["self_employed"])


X = df.drop("status", axis=1)
y = df["status"]


ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_resampled, test_size=0.2, random_state=42
)


knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train, y_train)


joblib.dump((knn_model, scaler, label_enc), "loan_model.pkl")
print("✅ Model trained and saved as loan_model.pkl")


knn_model, scaler, label_enc = joblib.load("loan_model.pkl")


sample_dict = {
    "loan_id": 101,                
    "dependents": 1,
    "education": 1,                
    "self_employed": 0,
    "imcome_annum": 1500000,
    "loan_amount": 500000,
    "loan_term": 12,
    "cibil_score": 780,
    "residential_assets_value": 800000,
    "commercial_assets_value": 500000,
    "luxury_assets_value": 100000,
    "bank_asset_value": 400000
}


if "loan_id" in sample_dict:
    del sample_dict["loan_id"]


sample = np.array([list(sample_dict.values())])
sample_scaled = scaler.transform(sample)
prediction = knn_model.predict(sample_scaled)

print("\nPrediction:", "Approved ✅" if prediction[0] == 1 else "Rejected ❌")
