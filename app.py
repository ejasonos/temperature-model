from flask import Flask, render_template, request, jsonify
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
# HF API
# =========================
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not set in environment variables")

API_URL = "https://api-inference.huggingface.co/models/gpt2"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def query(prompt):
    try:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 50,
                "temperature": 0.7
            }
        }

        response = requests.post(
            API_URL,
            headers=HEADERS,
            json=payload,
            timeout=30
        )

        # DEBUG
        print("STATUS:", response.status_code)
        print("RAW TEXT:", response.text[:200])

        # Handle empty response
        if not response.text.strip():
            return "Error: Empty response from API"

        # Try parsing JSON safely
        try:
            data = response.json()
        except Exception:
            return f"Non-JSON response: {response.text[:100]}"

        # Handle HF errors
        if isinstance(data, dict):
            return f"HF Error: {data.get('error', data)}"

        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"].replace(prompt, "").strip()

        return str(data)

    except requests.exceptions.RequestException as e:
        return f"Request failed: {str(e)}"

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
        response = query(prompt)

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)