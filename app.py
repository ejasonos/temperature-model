from flask import Flask, request, render_template
import numpy as np
import torch
import pickle

from model import TemperatureNN

app = Flask(__name__)

# =========================
# LOAD MODEL + SCALERS
# =========================
try:
    # Load PyTorch model
    model = TemperatureNN()
    model.load_state_dict(torch.load("temperature_model.pth", map_location="cpu"))
    model.eval()

    # Load scalers using open() + pickle
    with open("scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)

    with open("scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)

except Exception as e:
    model = None
    scaler_X = None
    scaler_y = None
    print("Error loading model/scalers:", e)


# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return render_template("index.html", prediction_text="Model not loaded")

    try:
        # =========================
        # GET INPUTS (must match form order)
        # =========================
        humidity = float(request.form["humidity"])
        windspeed = float(request.form["windspeed"])
        rainfall = float(request.form["rainfall"])

        features = np.array([[humidity, windspeed, rainfall]])

        # =========================
        # SCALE INPUT
        # =========================
        features_scaled = scaler_X.transform(features)

        # =========================
        # PREDICT WITH PYTORCH MODEL
        # =========================
        tensor_input = torch.tensor(features_scaled, dtype=torch.float32)

        with torch.no_grad():
            scaled_output = model(tensor_input).numpy()

        # =========================
        # INVERSE SCALE OUTPUT
        # =========================
        prediction = scaler_y.inverse_transform(scaled_output)

        temp = prediction[0][0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted Temperature: {round(temp, 2)} °C"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)
