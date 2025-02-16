from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("price_model.pkl")


@app.route("/")
def home():
    return "📢 Mobile Price Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json 
    features = np.array(data['features']).reshape(1, -1)  
    prediction = model.predict(features)  
    return jsonify({'predicted_price': prediction[0][0]})  

if __name__ == '__main__':
    app.run(debug=True) 