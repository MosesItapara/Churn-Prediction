from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import sys
import os

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_datapoint():
    try:
        # Create CustomData object from form data
        data = CustomData(
            CreditScore=int(request.form.get('CreditScore')),
            Geography=request.form.get('Geography'),
            Gender=request.form.get('Gender'),
            Age=int(request.form.get('Age')),
            Tenure=int(request.form.get('Tenure')),
            Balance=float(request.form.get('Balance')),
            NumOfProducts=int(request.form.get('NumOfProducts')),
            HasCrCard=int(request.form.get('HasCrCard')),
            IsActiveMember=int(request.form.get('IsActiveMember')),
            EstimatedSalary=float(request.form.get('EstimatedSalary'))
        )

        # Get prediction
        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(pred_df)

        # Return prediction result
        return render_template('result.html', prediction=prediction[0])

    except Exception as e:
        raise CustomException(e, sys)
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)