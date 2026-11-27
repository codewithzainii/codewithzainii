# Customer Churn Prediction

A machine learning system that predicts whether a telecom-style customer will churn (leave the service). A Random Forest model is trained on customer data and served through a Flask REST API. A console client sends customer details to the API and displays the churn risk.

## Architecture

```
console.py  ──HTTP POST──►  app2.py (Flask API)  ──►  model/ (Random Forest, scaler, encoders)
```

- **MLProjChurn.py**: loads the dataset, encodes categorical columns, scales features, trains the model, evaluates it, and saves the artifacts to `model/`.
- **app2.py**: Flask server with a `POST /predict` endpoint. It loads the saved artifacts and returns the prediction and probabilities.
- **console.py**: interactive command-line client that collects customer details and calls the API.

## Dataset

`customer_churn_dataset.csv` (about 2.8 MB) contains customer records with these features:

| Feature | Type |
|---|---|
| Age | numeric |
| Gender | categorical (Male, Female) |
| Tenure | numeric (months) |
| Usage Frequency | numeric |
| Support Calls | numeric |
| Payment Delay | numeric (days) |
| Subscription Type | categorical (Basic, Standard, Premium) |
| Contract Length | categorical (Monthly, Quarterly, Annual) |
| Total Spend | numeric |
| Last Interaction | numeric (days) |

Target: `Churn` (0 = stays, 1 = leaves).

## Method

1. Label-encode the three categorical columns.
2. Standardize all features with `StandardScaler`.
3. 80/20 train/test split (`random_state=42`).
4. Train `RandomForestClassifier(n_estimators=100, random_state=42)`.
5. Save the model, scaler, encoders, and feature order with joblib.

## Results

Test accuracy: **[ACCURACY]%**

Run `python MLProjChurn.py` to see the full classification report and confusion matrix.

## How to run

```bash
pip install -r requirements.txt
python MLProjChurn.py      # optional: retrain and regenerate model/
python app2.py             # start the API at http://127.0.0.1:5000
```

In a second terminal:

```bash
python console.py
```

## API

`POST /predict` with JSON:

```json
{
  "Age": 35, "Gender": "Male", "Tenure": 24,
  "Usage Frequency": 10, "Support Calls": 2, "Payment Delay": 5,
  "Subscription Type": "Standard", "Contract Length": "Monthly",
  "Total Spend": 500, "Last Interaction": 15
}
```

Response:

```json
{
  "prediction": 1,
  "result": "Will Churn ⚠️",
  "churn_probability": "72.4%",
  "safe_probability": "27.6%"
}
```

## Project structure

```
├── MLProjChurn.py
├── app2.py
├── console.py
├── customer_churn_dataset.csv
├── requirements.txt
└── model/
    ├── churn_model.pkl
    ├── encoders.pkl
    ├── feature_cols.pkl
    └── scaler.pkl
```

## Built with

Python · pandas · scikit-learn · Flask · joblib · requests
