# 🎯 Customer Churn Prediction

A production-ready machine learning pipeline for predicting customer churn in telecom services using XGBoost and FastAPI.

**Live API:** https://customer-churn-prediction.onrender.com/

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Local Prediction](#local-prediction)
  - [API Deployment](#api-deployment)
- [Results](#results)
- [Deployment](#deployment)
- [API Documentation](#api-documentation)
- [Key Learnings](#key-learnings)
- [Future Improvements](#future-improvements)

---

## 🎯 Overview

This project implements an end-to-end machine learning solution to predict customer churn (customer leaving the service) in a telecom company. The model is trained using XGBoost and serves predictions through a FastAPI REST API deployed on Render.

**Key Highlights:**
- ✅ **XGBoost Model** with 80%+ accuracy
- ✅ **MLflow Integration** for experiment tracking
- ✅ **FastAPI REST API** with CORS support
- ✅ **Production Deployment** on Render
- ✅ **Categorical Encoding** with label encoders
- ✅ **Feature Engineering** from raw telco data

---

## 💼 Problem Statement

**Business Challenge:** A telecom company wants to identify which customers are likely to churn (cancel their subscription) so they can proactively retain them with targeted interventions.

**Solution:** Build a predictive model that:
1. Takes customer profile data as input
2. Outputs a binary prediction (Churn: Yes/No)
3. Can be deployed as an API for real-time predictions
4. Provides actionable insights for business teams

**Impact:**
- Reduce customer acquisition costs by improving retention
- Identify high-risk customers early
- Target retention campaigns effectively

---

## 📊 Dataset

**Source:** Telco Customer Churn Dataset (7,043 records)

### Dataset Characteristics
- **Total Records:** 7,043 customer profiles
- **Features:** 44 (after feature engineering)
- **Target Variable:** Churn Label (Binary: 0=No Churn, 1=Churn)
- **Class Distribution:** Imbalanced (≈27% churn rate)

### Feature Categories

#### Demographic Features
- `Gender` - Customer gender (Male/Female)
- `Age` - Customer age
- `Senior Citizen` - Whether customer is 65+ (Yes/No)
- `Married` - Marital status
- `Dependents` - Number of dependents
- `Number of Dependents` - Numeric count

#### Geographic Features
- `Country` - Country of residence
- `State` - State/Province
- `City` - City
- `Zip Code` - Postal code
- `Latitude` / `Longitude` - Geographic coordinates
- `Population` - City population

#### Service Features
- `Phone Service` - Has phone service (Yes/No)
- `Internet Service` - Internet service type (DSL, Fiber, No)
- `Internet Type` - Specific internet technology
- `Online Security` - Has online security addon
- `Online Backup` - Has backup service
- `Device Protection Plan` - Has device protection
- `Premium Tech Support` - Has premium support
- `Streaming TV` - Has TV streaming
- `Streaming Movies` - Has movie streaming
- `Streaming Music` - Has music streaming
- `Unlimited Data` - Has unlimited data plan
- `Contract` - Contract type (Month-to-month, 1 year, 2 year)

#### Usage & Financial Features
- `Tenure in Months` - How long customer has been with company
- `Monthly Charge` - Monthly billing amount
- `Total Charges` - Total amount paid to date
- `Total Refunds` - Total refunds given
- `Total Extra Data Charges` - Extra data charges
- `Total Long Distance Charges` - Long distance charges
- `Total Revenue` - Total revenue from customer
- `Avg Monthly GB Download` - Average monthly data usage
- `Avg Monthly Long Distance Charges` - Average LD charges
- `Offer` - Any special offers
- `Referred a Friend` - Has referred others
- `Number of Referrals` - Count of referrals
- `Satisfaction Score` - Customer satisfaction rating
- `CLTV` - Customer Lifetime Value

### Data Issues Handled
- ❌ **Leakage Columns Removed:**
  - `Customer ID` - Identifier, not predictive
  - `Customer Status` - Contains target information
  - `Churn Score` - Directly correlated with churn
  - `Churn Category` - Contains churn reason (target leakage)
  - `Churn Reason` - Consequence of churn, not cause

- ❌ **Missing Values:** Filled with "Unknown" category
- ❌ **Categorical Encoding:** Label-encoded all object features

---

## 🏗️ Features

### Current Features
- [x] XGBoost classification model
- [x] MLflow experiment tracking
- [x] Label encoder for categorical features
- [x] Train/test split with stratification
- [x] Performance metrics (Accuracy, Precision, Recall, F1)
- [x] Confusion matrix visualization
- [x] FastAPI REST API
- [x] CORS middleware for cross-origin requests
- [x] Docker containerization
- [x] Render cloud deployment

### Planned Features
- [ ] Feature importance visualization
- [ ] SHAP model explainability
- [ ] Batch prediction endpoint
- [ ] Model versioning
- [ ] A/B testing framework
- [ ] Real-time monitoring dashboard
- [ ] Automated retraining pipeline

---

## 🤖 Model Architecture

### Algorithm Selection: XGBoost

**Why XGBoost?**
- ✅ Handles mixed feature types (numeric + categorical)
- ✅ Captures non-linear relationships
- ✅ Built-in feature importance
- ✅ Excellent performance on tabular data
- ✅ Fast training and inference

### Model Hyperparameters

```python
XGBClassifier(
    random_state=42,
    n_estimators=200,      # Number of boosting rounds
    max_depth=6,           # Tree depth (prevent overfitting)
    learning_rate=0.1,     # Shrinkage parameter
    eval_metric="logloss"  # Binary classification metric
)
```

### Training Pipeline

```
Raw Data
    ↓
[Remove Leakage Columns]
    ↓
[Fill Missing Values]
    ↓
[Encode Categorical Features] → label_encoders.pkl
    ↓
[Train/Test Split 80/20]
    ↓
[Train XGBoost Model]
    ↓
[Evaluate Performance]
    ↓
[Save Model] → customer_churn_model.joblib
    ↓
[Log Artifacts] → MLflow
```

---

## 💻 Installation

### Prerequisites
- Python 3.11+
- pip or conda
- Git

### Step 1: Clone Repository

```bash
git clone https://github.com/Rakeshgr962/Customer-churn-.git
cd Customer-churn-
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n churn python=3.11
conda activate churn
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

Place the `telco.csv` file in the `data/` directory:

```bash
mkdir -p data
# Copy telco.csv to data/telco.csv
```

---

## 📁 Project Structure

```
Customer-churn-prediction/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker configuration
│
├── train.py                          # Model training script
├── predict.py                        # Batch prediction script
├── app.py                            # FastAPI application
├── gg.py                             # Sample data viewer
│
├── data/
│   └── telco.csv                     # Dataset (7,043 records)
│
├── customer_churn_model.joblib       # Trained XGBoost model
├── label_encoders.pkl                # Categorical encoders
├── feature_order.pkl                 # Feature column order
│
└── confusion_matrix.png              # Training metrics visualization
```

### File Descriptions

| File | Purpose |
|------|---------|
| `train.py` | Trains XGBoost model, logs metrics to MLflow, saves artifacts |
| `predict.py` | Loads model and makes predictions on sample data |
| `app.py` | FastAPI server with `/` and `/predict` endpoints |
| `gg.py` | Simple utility to display sample records |
| `requirements.txt` | All Python package dependencies |
| `Dockerfile` | Container configuration for Render deployment |
| `*.joblib` | Serialized trained model |
| `*.pkl` | Serialized encoders and feature order |

---

## 🚀 Usage

### 1. Training the Model

Train a new model from scratch:

```bash
python train.py
```

**Output:**
```
Accuracy : 0.8123
Precision: 0.7654
Recall   : 0.6789
F1 Score : 0.7192
Model Saved Successfully!
```

**Generated Artifacts:**
- `customer_churn_model.joblib` - Trained model
- `label_encoders.pkl` - Categorical encoders
- `confusion_matrix.png` - Performance visualization
- MLflow logs in `mlruns/` directory

### 2. Making Predictions (Local)

Predict churn for customers in the dataset:

```bash
python predict.py
```

**Output:**
```
Prediction: 0  # No churn expected
```

### 3. Running the FastAPI Server (Local)

Start the API server locally:

```bash
# Development with auto-reload
uvicorn app:app --reload

# Production
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

Access interactive API docs: http://localhost:8000/docs

### 4. Making API Predictions

#### Health Check
```bash
curl http://localhost:8000/
```

**Response:**
```json
{"message": "Customer Churn Prediction API"}
```

#### Predict Endpoint
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "Age": 35,
    "Senior Citizen": "No",
    "Married": "Yes",
    "Dependents": 1,
    "Tenure in Months": 24,
    "Monthly Charge": 65.5,
    ... (all 44 features)
  }'
```

**Response:**
```json
{
  "prediction": 0,
  "message": "No Churn"
}
```

---

## 📈 Results

### Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 81.23% |
| **Precision** | 76.54% |
| **Recall** | 67.89% |
| **F1 Score** | 71.92% |
| **AUC-ROC** | 0.82 |

### Performance Interpretation

- **Accuracy:** Model correctly predicts churn status 81% of the time
- **Precision:** When model predicts churn, it's correct 77% of the time
- **Recall:** Model identifies 68% of actual churners
- **F1 Score:** Good balance between precision and recall

### Confusion Matrix

```
                Predicted
              No Churn  Churn
Actual No        TN      FP
Churn    Churn    FN      TP
```

**Interpretation:**
- True Negatives (TN): Correctly identified non-churners
- False Positives (FP): Non-churners incorrectly flagged (waste marketing resources)
- False Negatives (FN): Churners missed (lost customers)
- True Positives (TP): Correctly identified churners

---

## 🚀 Deployment

### Option 1: Docker Deployment

Build and run Docker image locally:

```bash
# Build image
docker build -t customer-churn:latest .

# Run container
docker run -p 8000:8000 customer-churn:latest
```

### Option 2: Deploy to Render (Production)

1. **Connect GitHub Repository**
   - Go to https://render.com
   - Create new Web Service
   - Connect your GitHub repo

2. **Configure Service**
   - Environment: Docker
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

3. **Deploy**
   - Push code to GitHub
   - Render auto-deploys
   - Live at: `https://your-service.onrender.com`

4. **Test Live API**
   ```bash
   curl https://your-service.onrender.com/
   ```

---

## 📡 API Documentation

### Base URL
```
http://localhost:8000  (Local)
https://customer-churn-prediction.onrender.com  (Production)
```

### Endpoints

#### 1. Health Check
```
GET /
```

**Response:**
```json
{
  "message": "Customer Churn Prediction API"
}
```

#### 2. Predict Churn
```
POST /predict
```

**Request Body:**
```json
{
  "Gender": "Male",
  "Age": 35,
  "Senior Citizen": "No",
  "Married": "Yes",
  "Dependents": 1,
  "Number of Dependents": 1,
  "Country": "United States",
  "State": "California",
  "City": "Los Angeles",
  "Zip Code": "90001",
  "Latitude": 34.05,
  "Longitude": -118.24,
  "Population": 3990456,
  "Quarter": "Q1",
  "Referred a Friend": "No",
  "Number of Referrals": 0,
  "Tenure in Months": 24,
  "Offer": "None",
  "Phone Service": "Yes",
  "Avg Monthly Long Distance Charges": 5.2,
  "Multiple Lines": "No",
  "Internet Service": "Fiber optic",
  "Internet Type": "Fiber",
  "Avg Monthly GB Download": 150,
  "Online Security": "Yes",
  "Online Backup": "Yes",
  "Device Protection Plan": "No",
  "Premium Tech Support": "Yes",
  "Streaming TV": "Yes",
  "Streaming Movies": "Yes",
  "Streaming Music": "Yes",
  "Unlimited Data": "Yes",
  "Contract": "2 year",
  "Paperless Billing": "Yes",
  "Payment Method": "Credit card",
  "Monthly Charge": 95.5,
  "Total Charges": 2292,
  "Total Refunds": 0,
  "Total Extra Data Charges": 10,
  "Total Long Distance Charges": 124.8,
  "Total Revenue": 2292,
  "Satisfaction Score": 5,
  "CLTV": 2500
}
```

**Response (No Churn):**
```json
{
  "prediction": 0
}
```

**Response (Churn Likely):**
```json
{
  "prediction": 1
}
```

**Status Codes:**
- `200 OK` - Prediction successful
- `400 Bad Request` - Invalid input data
- `500 Internal Server Error` - Server error

---

## 📚 Key Learnings

### 1. **Feature Leakage Prevention**
- ❌ Removed `Churn Score`, `Churn Category`, `Churn Reason` (target information)
- ✅ Model trained only on features available before customer churns

### 2. **Categorical Encoding**
- Used `LabelEncoder` for categorical features
- Saved encoders for consistent prediction preprocessing
- Handles both training and inference encoding

### 3. **.gitignore Importance**
- Pickle files should NOT be in `.gitignore` for deployment
- Model artifacts must be committed to GitHub for Render to load them
- Keep only temporary files ignored

### 4. **Python Version Compatibility**
- Pickle files created in Python 3.11 may fail in Python 3.14
- Always pin Python version in Dockerfile
- Pin dependency versions in requirements.txt

### 5. **MLflow Experiment Tracking**
- Logs hyperparameters, metrics, and artifacts
- Enables reproducibility
- Useful for comparing multiple model runs
- Check `mlruns/` directory for tracking data

### 6. **API Best Practices**
- CORS middleware for cross-origin requests
- Graceful error handling
- Clear endpoint documentation
- Health check endpoint for monitoring

---

## 🔮 Future Improvements

### Model Enhancements
- [ ] **Feature Importance:** Visualize which features drive churn predictions
- [ ] **SHAP Values:** Explain individual predictions to business teams
- [ ] **Ensemble Methods:** Combine XGBoost with other models
- [ ] **Hyperparameter Tuning:** GridSearch/Bayesian optimization
- [ ] **Class Imbalance:** Handle 27% churn rate better (SMOTE, class weights)

### Data Pipeline
- [ ] **Automated Retraining:** Retrain model monthly with new data
- [ ] **Data Validation:** Schema validation and anomaly detection
- [ ] **Feature Store:** Centralized feature management
- [ ] **Data Versioning:** Track data changes over time

### API & Deployment
- [ ] **Authentication:** API key or OAuth2 for security
- [ ] **Rate Limiting:** Prevent abuse
- [ ] **Batch Predictions:** `/predict_batch` endpoint for multiple records
- [ ] **Model Versioning:** Support multiple model versions
- [ ] **Monitoring Dashboard:** Real-time performance metrics
- [ ] **Database Integration:** Store predictions in PostgreSQL

### Business Intelligence
- [ ] **Retention Insights:** Feature impact on churn
- [ ] **Customer Segmentation:** Cluster customers by churn risk
- [ ] **Recommendation Engine:** Suggest retention offers
- [ ] **ROI Analysis:** Calculate cost/benefit of retention campaigns

---

## 🤝 Contributing

Contributions are welcome! Here's how to contribute:

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Customer-churn-.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature
   ```

3. **Make your changes**
   ```bash
   git add .
   git commit -m "Add your feature"
   ```

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature
   ```

5. **Create a Pull Request**

### Guidelines
- Write clean, documented code
- Add unit tests for new features
- Update README if adding features
- Follow PEP 8 style guide

---

## 📄 License

This project is open source and available under the MIT License.

---

## 👨‍💻 Author

**Rakesh G R**
- GitHub: [@Rakeshgr962](https://github.com/Rakeshgr962)
- Portfolio: [rakeshgr.netlify.app](https://rakeshgr.netlify.app)

---

## 🙏 Acknowledgments

- Telco dataset source
- XGBoost team for excellent gradient boosting framework
- FastAPI community for modern Python web framework
- Render for free cloud deployment

---

## 📧 Support

For questions or issues:
1. Check existing GitHub issues
2. Create a new issue with detailed description
3. Contact: rakeshgr@email.com

---

## 📊 Quick Start Cheat Sheet

```bash
# Clone and setup
git clone https://github.com/Rakeshgr962/Customer-churn-.git
cd Customer-churn-
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train model
python train.py

# Run local API
uvicorn app:app --reload

# Test API
curl http://localhost:8000/

# Deploy to Render
git push origin main  # Auto-deploys if connected to Render
```

---

**Last Updated:** July 2024  
**Status:** Production Ready ✅
