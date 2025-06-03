from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Charger le modèle une seule fois au démarrage
with open("modele_random_forest.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Serveur IA actif"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "features" not in data:
        return jsonify({"error": "Paramètre 'features' manquant"}), 400
    
    try:
        features = np.array(data["features"]).reshape(1, -1)  # 1 ligne, N colonnes
        prediction = model.predict(features)
        return jsonify({"prediction": str(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

