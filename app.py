from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Charger le modèle
model = joblib.load("modele_random_forest.pkl")

@app.route('/', methods=['GET'])
def index():
    return "Serveur IA en ligne."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get("data", [])
    
    if not data:
        return jsonify({"error": "Pas de données envoyées"}), 400

    try:
        data_np = np.array(data).reshape(1, -1)
        prediction = model.predict(data_np)[0]
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
