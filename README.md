# 🏦 Loan Approval Prediction using Neural Networks

A deep learning–based system to predict loan approval decisions using applicant financial and demographic data.  
Developed and maintained by **Surajit**.

---

## 🚀 Overview

This project uses a **PyTorch feed-forward neural network** to predict whether a loan application will be **Approved** or **Rejected**, based on various financial and credit-related features.

The workflow:
1. Load a trained model and preprocessing objects (scaler and label encoders).
2. Read applicant data from an Excel file (`loan_applications.xlsx`).
3. Preprocess the data (encode categorical columns and scale numerical features).
4. Generate predictions and save the results in `loan_applications_result.xlsx`.

---

## 🧠 Model Architecture

The neural network (`LoanNN`) consists of:
- Input Layer → `input_dim = 11`
- Hidden Layer 1 → 64 neurons + ReLU + Dropout(0.3)
- Hidden Layer 2 → 32 neurons + ReLU
- Output Layer → 1 neuron + Sigmoid activation

This structure outputs a probability score representing the likelihood of loan approval.

---

## 📁 Project Structure

```
LOAN_APPROVAL/
├── .dist/                         # Distribution or build artifacts
├── data/                          # Raw or processed data files
├── Loan/                          # Core loan-related modules
├── models/                        # Trained models and architectures
├── output_data/                   # Generated results or outputs
│
├── .gitignore                     # Git ignore file
├── loan_approval.ipynb            # Main notebook for model training/evaluation
├── nn_model.ipynb                 # Neural network model development notebook
├── nn_run_model.py                # Script to run the trained neural network model
├── run_model.py                   # Prediction or model execution script
├── xcel_file_test.py              # Excel file testing and validation script
│
├── requirements.txt               # List of dependencies
└── README.md                      # Project documentation
                      # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Surajit09636/loan-approval-predictor.git
cd loan-approval-predictor
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> If you don't have `requirements.txt` yet, generate it using:
> ```bash
> pip freeze > requirements.txt
> ```

---

## 📊 Input Data Format

The model expects an Excel file (`loan_applications.xlsx`) with the following columns:

| Column | Description |
|---------|--------------|
| dependents | Number of dependents |
| education | Graduate / Not Graduate |
| self_employed | Yes / No |
| imcome_annum | Annual income |
| loan_amount | Loan amount requested |
| loan_term | Term in months |
| cibil_score | Credit score |
| residential_assets_value | Value of residential assets |
| commercial_assets_value | Value of commercial assets |
| luxury_assets_value | Value of luxury assets |
| bank_asset_value | Value of bank assets |

Example:

| dependents | education | self_employed | imcome_annum | loan_amount | loan_term | cibil_score | residential_assets_value | commercial_assets_value | luxury_assets_value | bank_asset_value |
|-------------|------------|----------------|---------------|--------------|-------------|--------------|----------------------------|---------------------------|----------------------|------------------|
| 1 | Graduate | No | 600000 | 150000 | 24 | 720 | 250000 | 50000 | 20000 | 70000 |

---

## 🧩 Running Predictions

Run the main script:
```bash
python loan_predict.py
```

It will:
- Load your model (`./data/loan_nn_model_gpu.pth`)
- Process `loan_applications.xlsx`
- Generate approval predictions
- Save results to `loan_applications_result.xlsx`

Sample output:
```
✅ Predictions saved to 'loan_applications_result.xlsx'
```

---

## 🧾 Output Format

The output Excel file will include:
- `Prediction_Probability` → Model confidence score
- `Predicted_Status` → “Approved” or “Rejected”

---

## 🧰 Dependencies

Main Python libraries:
- `torch`
- `pandas`
- `scikit-learn`
- `openpyxl`
- `numpy`

You can install them with:
```bash
pip install torch pandas scikit-learn openpyxl numpy
```

---

## 👨‍💻 Developed By

**Surajit**  
📧 Email: [surajitsutradhar010@gmail.com](mailto:surajitsutradhar010@gmail.com)  
💼 GitHub: [https://github.com/Surajit09636](https://github.com/Surajit09636)

> *"Empowering smarter financial decisions through machine learning."*

---

## 📝 License

This project is released under the MIT License.  
You’re free to use, modify, and distribute with proper credit to **Surajit**.

---
