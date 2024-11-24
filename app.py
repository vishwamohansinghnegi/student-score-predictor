from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')  # Welcome page with link to form

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')  # Render the prediction form on GET request
    else:
        # Collecting form data for prediction
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),  # Fix to correctly map input fields
            writing_score=float(request.form.get('writing_score'))   # Fix to correctly map input fields
        )

        # Convert data to DataFrame for prediction
        pred_df = data.get_data_as_data_frame()
        print(pred_df)  # For debugging, printing the input data
        print("Before Prediction")

        # Using the predict pipeline to get results
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

        # Render the result on the same home.html page with the prediction result
        return render_template('home.html', results=results[0])


if __name__ == "__main__":
    print("Server is running on http://localhost:5000/")
    app.run(host="localhost", port=5000, debug=True)
