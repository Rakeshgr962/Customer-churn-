# Customer Churn Prediction

This project contains a simple machine learning pipeline for customer churn prediction.

## Structure
- data/ - dataset location
- models/ - trained model artifacts
- mlruns/ - MLflow tracking directory
- train.py - trains the model and logs metrics with MLflow
- predict.py - loads the trained model and predicts a single record
- app.py - Flask API for serving predictions

## Usage
1. Place the telco dataset at data/telco.csv
2. Install dependencies: pip install -r requirements.txt
3. Train the model: python train.py
4. Run the API: python app.py
