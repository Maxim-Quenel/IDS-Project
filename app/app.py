from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import os

app = Flask(__name__)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(APP_DIR)
MODEL_FILENAMES = ["model_pipeline.joblib", "model_catboost_pipeline.joblib"]

def resolve_model_path():
    search_dirs = [APP_DIR, PARENT_DIR]
    for search_dir in search_dirs:
        for filename in MODEL_FILENAMES:
            candidate = os.path.join(search_dir, filename)
            if os.path.isfile(candidate):
                return candidate
    searched = ", ".join(search_dirs)
    expected = ", ".join(MODEL_FILENAMES)
    raise FileNotFoundError(
        f"Model file not found. Looked in: {searched}. Expected one of: {expected}."
    )

# Charger le modele (pipeline)
model_path = resolve_model_path()
model_data = joblib.load(model_path)
pipeline = model_data['pipeline']
feature_cols = model_data['feature_cols']


@app.route("/")
def index():
    """Sert la page HTML du formulaire."""
    return send_from_directory(os.path.dirname(__file__), "index.html")


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint pour prédire normal/attack.
    """
    data = request.get_json()
    if data is None:
        return jsonify({"error": "No JSON payload provided"}), 400

    # Accepter soit un dict (1 sample) soit une liste de dicts
    if isinstance(data, dict):
        input_df = pd.DataFrame([data])
    elif isinstance(data, list):
        input_df = pd.DataFrame(data)
    else:
        return jsonify({"error": "Payload must be a dict or a list of dicts"}), 400

    # S'assurer que toutes les colonnes attendues sont présentes
    for col in feature_cols:
        if col not in input_df.columns:
            # mettre une valeur par défaut
            if col in ['protocol_type', 'service', 'flag']:
                input_df[col] = ""  # string vide pour les colonnes catégorielles
            else:
                input_df[col] = 0   # zéro pour les colonnes numériques


    # Garder l’ordre
    input_df = input_df[feature_cols]

    try:
        preds = pipeline.predict(input_df)
    except Exception as e:
        return jsonify({"error": "Prediction failed", "detail": str(e)}), 500

    # Map en chaînes
    out = ["normal" if int(p) == 0 else "attack" for p in preds]

    return jsonify({"predictions": out})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
