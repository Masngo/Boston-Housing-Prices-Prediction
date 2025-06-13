from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None  # Initialize prediction variable
    error = None  # Initialize error variable
    if request.method == 'POST':
        try:
            # Retrieve and convert form data, providing default values if empty
            features = [
                float(request.form.get('CRIM', 0) or 0),
                float(request.form.get('ZN', 0) or 0),
                float(request.form.get('INDUS', 0) or 0),
                float(request.form.get('CHAS', 0) or 0),
                float(request.form.get('NOX', 0) or 0),
                float(request.form.get('RM', 0) or 0),
                float(request.form.get('AGE', 0) or 0),
                float(request.form.get('LSTAT', 0) or 0)
            ]

            # Make prediction
            prediction = model.predict([features])[0]

            # Prevent negative predictions
            if prediction < 0:
                prediction = 0  # Set to zero or a minimum expected price

        except ValueError as e:
            error = f"Error: {e}"

    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)