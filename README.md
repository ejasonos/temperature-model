# 🌡️ Temperature Prediction Neural Network

This project is a Neural Network-based system designed to predict temperature using environmental inputs such as humidity, wind speed, and rainfall.

---

## 🚀 Overview

The model learns patterns from historical environmental data to estimate temperature values. It demonstrates how machine learning can model nonlinear relationships between weather variables.

---

## 🧠 Model Description

The system uses a **feedforward neural network** trained on structured data.

### Inputs:
- Humidity  
- Wind Speed  
- Rainfall  

### Output:
- Predicted Temperature  

---

## ⚙️ Features

- Neural network-based regression  
- Data preprocessing and normalization  
- Prediction from new environmental inputs  
- Lightweight and extendable design  

---

## 📁 Project Structure

~~~
temperature-model/
│── data.csv              # Dataset containing environmental variables and temperature
│── train.py              # Script for training the neural network model
│── predict.py            # Script for making temperature predictions
│── model.pkl             # Saved trained model
│── scaler.pkl            # Saved data scaler for preprocessing
│── requirements.txt      # Project dependencies
│── README.md             # Project documentation
~~~

---

## ▶️ Usage

1. Train the model using the training script  
2. Use the prediction script to input new values  
3. The model outputs the predicted temperature  

---

## 📊 How It Works

1. Load dataset  
2. Preprocess and scale input features  
3. Train neural network on data  
4. Save trained model  
5. Use model to predict temperature  

---

## 📌 Future Improvements

- Add more weather features (e.g., pressure, sunlight)  
- Improve model accuracy with tuning  
- Deploy as an API  
- Integrate real-time data sources  

---

## 📜 License

This project is open-source and available under the MIT License.