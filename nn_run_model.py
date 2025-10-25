import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load(
    "loan_nn_model_gpu.pth",
    map_location=device,
    weights_only=False  
)


input_dim = 11  
model = LoanNN(input_dim)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()


scaler = checkpoint["scaler"]
label_enc = checkpoint["label_enc"]


sample = pd.DataFrame([{
    "dependents": 3,
    "education": 1,
    "self_employed": 0,
    "imcome_annum": 1500000,
    "loan_amount": 25000000,
    "loan_term": 20,
    "cibil_score": 200,
    "residential_assets_value": 500000,
    "commercial_assets_value": 300000,
    "luxury_assets_value": 100000,
    "bank_asset_value": 200000
}])


sample["education"] = label_enc.transform(sample["education"])
sample["self_employed"] = label_enc.transform(sample["self_employed"])


X_sample = scaler.transform(sample)


X_tensor = torch.tensor(X_sample, dtype=torch.float32).to(device)


with torch.no_grad():
    y_pred_prob = model(X_tensor)
    y_pred = (y_pred_prob >= 0.5).float()

print(f"Prediction Probability: {y_pred_prob.item():.4f}")
print("Predicted Class:", "Approved" if y_pred.item() == 1 else "Rejected")
