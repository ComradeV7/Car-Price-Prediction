from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load model and data once on startup
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))
car = pd.read_csv("data/cleaned_cars.csv")

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = car.groupby('company')['name'].apply(lambda x: sorted(x.unique())).to_dict()
    years = sorted(car['year'].unique())
    fuel_types = sorted(car['fuel_type'].unique())
    companies.insert(0, 'Select Company')
    return render_template('index.html',
                           companies=companies,
                           car_models=car_models,
                           years=years,
                           fuel_types=fuel_types)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        company = request.form.get('company')
        car_model = request.form.get('car-model')
        year = request.form.get('year')
        fuel_type = request.form.get('fuel-type')
        km_driven_str = request.form.get('kilo-driven')

        # Basic validation and conversions
        if not all([company, car_model, year, fuel_type, km_driven_str]):
            return "Error: Missing input fields", 400

        try:
            km_driven = int(km_driven_str)
            year = int(year)
        except ValueError:
            return "Error: Year and Km driven must be integers", 400

        # Debug: (replace with logger if needed)
        # print(f"Received data: {company}, {car_model}, {year}, {fuel_type}, {km_driven}")

        # Prepare input dataframe for prediction (match feature order/names expected by model)
        input_df = pd.DataFrame({
            'company': [company],
            'name': [car_model],
            'year': [year],
            'fuel_type': [fuel_type],
            'kms_driven': [km_driven]
        })

        prediction = model.predict(input_df)

        # Return rounded prediction as string
        return f"{np.round(prediction[0], 2)}"

    except Exception as e:
        # For debugging in dev; in production return generic message
        return f"Error during prediction: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
