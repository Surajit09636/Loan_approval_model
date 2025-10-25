import torch
import torch.nn as nn
import pandas as pd
import openpyxl

# === Define the same model architecture ===
class LoanNN(nn.Module):
    def __init__(self, input_dim):
        super(LoanNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# === Load model, scaler, and label encoder ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load(
    "./models/loan_nn_model_gpu.pth",
    map_location=device,
    weights_only=False
)

input_dim = 11  # Number of features
model = LoanNN(input_dim)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

scaler = checkpoint["scaler"]
label_enc = checkpoint["label_enc"]

# === Load Excel file with multiple applications ===
file_path = "./data/loan_applications.xlsx"  # Your input Excel file
df = pd.read_excel(file_path)

# Make sure all required columns exist
required_cols = [
    "dependents", "education", "self_employed", "imcome_annum", "loan_amount", 
    "loan_term", "cibil_score", "residential_assets_value", 
    "commercial_assets_value", "luxury_assets_value", "bank_asset_value"
]
df = df[required_cols]

# Encode categorical features
df["education"] = label_enc.transform(df["education"])
df["self_employed"] = label_enc.transform(df["self_employed"])

# Scale numerical features
X_scaled = scaler.transform(df)

# Convert to tensor and move to device
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

# Predict
with torch.no_grad():
    y_pred_prob = model(X_tensor)
    y_pred = (y_pred_prob >= 0.5).float()

# Add predictions to dataframe
df["Prediction_Probability"] = y_pred_prob.cpu().numpy()
df["Predicted_Status"] = ["Approved" if x == 1 else "Rejected" for x in y_pred.cpu().numpy()]

# Save results to a new Excel file
output_file = "./output_data/loan_applications_result.xlsx"
df.to_excel(output_file, index=False)

print(f" Predictions saved to '{output_file}'")
