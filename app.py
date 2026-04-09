from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained regression model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        humidity = float(request.form['humidity'])
        windspeed = float(request.form['windspeed'])
        rainfall = float(request.form['rainfall'])

        # Arrange in same order used during training
        features = np.array([[humidity, windspeed, rainfall]])

        # Predict temperature
        prediction = model.predict(features)

        output = round(prediction[0], 2)

        return render_template(
            'index.html',
            prediction_text=f"Predicted Temperature: {output} °C"
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)
