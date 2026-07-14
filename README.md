# 🎯 Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn using **XGBoost**. The model is deployed as a **FastAPI REST API** on Render.

**🔗 Live API:** https://customer-churn-72v1.onrender.com

## 🚀 Features

* XGBoost Classification Model
* FastAPI REST API
* MLflow Experiment Tracking
* Label Encoding for Categorical Data
* Docker Support
* Render Deployment

## 📊 Dataset

* **Dataset:** Telco Customer Churn
* **Records:** 7,043
* **Target:** Churn Label (0 = No, 1 = Yes)

### Preprocessing

* Removed leakage columns:

  * Customer ID
  * Customer Status
  * Churn Score
  * Churn Category
  * Churn Reason
* Filled missing values
* Encoded categorical features

## 🤖 Model

* Algorithm: **XGBoost**
* Train/Test Split: **80/20**
* Experiment Tracking: **MLflow**

## 📁 Project Structure

```text
Customer-churn-prediction/
│── app.py
│── train.py
│── predict.py
│── requirements.txt
│── Dockerfile
│── customer_churn_model.joblib
│── label_encoders.pkl
│── feature_order.pkl
└── data/telco.csv
```

## ▶️ Run Locally

```bash
git clone https://github.com/Rakeshgr962/Customer-churn-.git
cd Customer-churn-
pip install -r requirements.txt

python train.py
uvicorn app:app --reload
```

Open:

```
http://localhost:8000/docs
```

## 📈 Results

| Metric    | Score |
| --------- | ----: |
| Accuracy  |  ~81% |
| Precision |  ~77% |
| Recall    |  ~68% |
| F1 Score  |  ~72% |

## 🌐 API

### Health Check

```
GET /
```

### Predict

```
POST /predict
```

Returns:

```json
{
  "prediction": 0
}
```

## 🚀 Deployment

* FastAPI
* Docker
* Render

## 🛠 Tech Stack

* Python
* XGBoost
* FastAPI
* MLflow
* Pandas
* Scikit-learn
* Docker

## 📌 Future Improvements

* SHAP Explainability
* Batch Prediction API
* Model Monitoring
* Automated Retraining

## 👨‍💻 Author

**Rakesh G R**

* GitHub: https://github.com/Rakeshgr962
* Portfolio: https://rakeshgr.netlify.app
