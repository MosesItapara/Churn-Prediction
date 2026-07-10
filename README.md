# Bank Customer Churn Predictor

A Flask web app that predicts whether a bank customer is likely to churn (leave the bank), based on a scikit-learn / XGBoost classification pipeline trained on the classic [Churn Modelling](https://www.kaggle.com/datasets/shubh0799/churn-modelling) dataset.

The UI lets you set a customer profile with sliders and toggles and get a live churn-risk prediction with an animated probability gauge — no page reload.

## Features

- End-to-end ML pipeline: ingestion → preprocessing → training → inference
- Model selection between Logistic Regression and XGBoost by test accuracy
- Flask backend exposing a JSON `/predict` API
- Dynamic, single-page prediction UI (sliders, pill toggles, animated result gauge)
- Custom exception handling and logging throughout the pipeline

## Project Structure

```
CHURN/
├── app.py                      # Flask app: serves the UI and /predict API
├── templates/
│   └── index.html              # Prediction UI
├── notebook/
│   ├── data/churn2.csv         # Source dataset
│   ├── EDA CHURN.ipynb         # Exploratory data analysis
│   └── MODELLING.ipynb         # Model experimentation
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Reads raw data, splits train/test
│   │   ├── data_transformation.py  # Builds the preprocessing pipeline
│   │   └── model_trainer.py        # Trains and selects the best model
│   ├── pipeline/
│   │   ├── train_pipeline.py       # Runs ingestion → transform → train
│   │   └── predict_pipeline.py     # Loads artifacts and serves predictions
│   ├── exception.py             # Custom exception with traceback context
│   ├── logger.py                # Logging configuration
│   └── utils.py                 # save_object / load_object / evaluate_model
├── artifacts/                   # Generated: train/test splits, model.pkl, preprocessor.pkl
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/MosesItapara/Churn-Prediction.git
cd Churn-Prediction

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

## Training the model

Generates `artifacts/model.pkl` and `artifacts/preprocessor.pkl`:

```bash
python -m src.pipeline.train_pipeline
```

This ingests `notebook/data/churn2.csv`, splits it into train/test sets, fits a `StandardScaler` on the numeric features, trains both a Logistic Regression and an XGBoost classifier, and saves whichever scores higher on test accuracy.

## Running the app

```bash
python app.py
```

Visit `http://localhost:5000`, set a customer profile, and submit to see the churn prediction.

### API

`POST /predict` accepts form data with the following fields and returns JSON:

| Field | Type | Notes |
|---|---|---|
| `CreditScore` | int | 350–850 |
| `Geography` | string | `France`, `Germany`, or `Spain` |
| `Gender` | string | `Male` or `Female` |
| `Age` | int | |
| `Tenure` | int | years with the bank |
| `Balance` | float | |
| `NumOfProducts` | int | 1–4 |
| `HasCrCard` | 0 or 1 | |
| `IsActiveMember` | 0 or 1 | |
| `EstimatedSalary` | float | |

Response:

```json
{ "churn": false, "probability": 0.17 }
```

## Tech Stack

- Python, Flask
- scikit-learn, XGBoost
- pandas, numpy
- Vanilla HTML/CSS/JS on the frontend (no build step)
