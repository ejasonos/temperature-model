from flask import Flask, render_template, request, jsonify
from transformers import pipeline

import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import traceback
import math

# Load GPT-2 model
generator = pipeline("text-generation", model="gpt2")

app = Flask(__name__)

# =========================
# MODEL (INLINE - NO model.py)
# =========================
class TemperatureNN(nn.Module):
    def __init__(self):
        super(TemperatureNN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# FEATURE CONFIG (CLEAN & SAFE)
# =========================
FEATURES = ["humidity", "windspeed", "rainfall"]


# =========================
# LOAD MODEL + SCALERS
# =========================
try:
    model = TemperatureNN()
    model.load_state_dict(torch.load("temperature_model.pth", map_location="cpu"))
    model.eval()

    with open("scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)

    with open("scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)
    
    print("Model and scalers loaded successfully")

except Exception as e:
    print("FULL ERROR TRACE:")
    print(traceback.format_exc())
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
        # CLEAN FEATURE EXTRACTION
        # =========================
        features = np.array([
            [float(request.form[key]) for key in FEATURES]
        ])

        # =========================
        # SCALE INPUT
        # =========================
        features_scaled = scaler_X.transform(features)

        # =========================
        # CONVERT TO TENSOR
        # =========================
        tensor_input = torch.tensor(features_scaled, dtype=torch.float32)

        # =========================
        # PREDICTION
        # =========================
        with torch.no_grad():
            scaled_output = model(tensor_input).numpy()

        # =========================
        # INVERSE SCALE OUTPUT
        # =========================
        prediction = scaler_y.inverse_transform(scaled_output)

        temp = prediction[0][0]
        truncate_temp = math.trunc(temp * 10) / 10

        return render_template(
            "index.html",
            prediction_text=f"Predicted Temperature: {truncate_temp}°C"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.form["prompt"]

    output = generator(
        prompt,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.7
    )

    text = output[0]["generated_text"]
    response = text.replace(prompt, "").strip()

    return render_template("index.html", response=response)

# =========================
# MAIN ENTRY
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True,host="0.0.0.0", port=port)
