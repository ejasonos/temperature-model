from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient

import requests
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import traceback
import math

app = Flask(__name__)

# =========================
# HF setup
# =========================
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = os.getenv("API_URL")
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# =========================
# MODEL
# =========================
class TemperatureNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


FEATURES = ["humidity", "windspeed", "rainfall"]


# =========================
# LOAD MODEL
# =========================
try:
    model = TemperatureNN()
    model.load_state_dict(torch.load("temperature_model.pth", map_location="cpu"))
    model.eval()

    with open("scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)

    with open("scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)

    print("Model loaded successfully")

except Exception:
    print(traceback.format_exc())
    model = None
    scaler_X = None
    scaler_y = None


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
        features = np.array([[
            float(request.form.get(key, 0)) for key in FEATURES
        ]])

        features_scaled = scaler_X.transform(features)
        tensor_input = torch.tensor(features_scaled, dtype=torch.float32)

        with torch.no_grad():
            scaled_output = model(tensor_input).numpy()

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


# =========================
# JSON GENERATION (FIXED)
# =========================
@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()

        if not data or "prompt" not in data:
            return jsonify({"error": "Missing prompt"}), 400

        prompt = data["prompt"]
        print(f"Prompt from frontend: {prompt}")

        payload = {"inputs": prompt
        response = requests.post(API_URL, headers=HEADERS, json=payload)

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)