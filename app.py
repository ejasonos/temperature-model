from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model safely
try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
except Exception as e:
    model = None
    print("Error loading model:", e)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Convert inputs to float (better than int for temperature data)
        features = [float(x) for x in request.form.values()]
        final_features = np.array([features])

        prediction = model.predict(final_features)

        return render_template(
            "index.html",
            prediction_text=f"Predicted Temperature: {round(prediction[0], 2)}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)
