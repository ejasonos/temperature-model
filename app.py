from flask import Flask, render_template, request, jsonify
from openai import OpenAI

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
NVIDIA_TOKEN = os.getenv("NVIDIA_TOKEN")
BASE_URL = os.getenv("BASE_URL")

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

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()

        if not data or "prompt" not in data:
            return jsonify({"error": "Missing prompt"}), 400

        prompt = f'Strictly Respond in less than 40 words. Question: As an Electric Vehicle analyst {data["prompt"]}'
        print(f"Prompt from frontend: {prompt}")

        client = OpenAI(
  api_key=NVIDIA_TOKEN,
  base_url=BASE_URL)

        try:
            response = client.chat.completions.create(
            model="minimaxai/minimax-m2.7",
            messages=[
                {"role": "user", "content": prompt}
            ])

            output = response.choices[0].message.content

        except Exception as e:
            output = f"Error: {str(e)}"

        return output

    except Exception as e:
        return f"Failed: {str(e)}"


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    app.run(debug=True)