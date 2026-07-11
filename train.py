import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

from xgboost import XGBClassifier

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

# Fill missing values
df.fillna("Unknown", inplace=True)

# Encode categorical columns
encoders = {}

for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Save encoders
joblib.dump(encoders, "label_encoders.pkl")

# Features & Target
X = df.drop("Churn Label", axis=1)
y = df["Churn Label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )
}

mlflow.set_experiment("Customer_Churn")

best_accuracy = 0

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, pred)

        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", accuracy)

        disp = ConfusionMatrixDisplay.from_predictions(y_test, pred)

        plt.savefig("confusion_matrix.png")
        plt.close()

        mlflow.log_artifact("confusion_matrix.png")

        mlflow.sklearn.log_model(model, "model")

        print(name, accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            joblib.dump(model, "customer_churn_model.joblib")

print("Best Model Saved")