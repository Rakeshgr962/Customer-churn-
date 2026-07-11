import joblib
import pandas as pd

# Load model
model = joblib.load("customer_churn_model.joblib")

# Load dataset
df = pd.read_csv("data/telco.csv")

# Remove leakage columns
df.drop(columns=[
    "Customer ID",
    "Customer Status",
    "Churn Score",
    "Churn Category",
    "Churn Reason"
], inplace=True)

# Remove target
X = df.drop("Churn Label", axis=1)

# Encode categorical columns
from sklearn.preprocessing import LabelEncoder

for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Predict first customer
sample = X.iloc[[0]]

prediction = model.predict(sample)

print("Prediction:", prediction[0])