from pathlib import Path

import joblib
import pandas as pd

from flask import Flask
from flask import render_template
from flask import request


# ==========================================================
# FLASK APP
# ==========================================================

app = Flask(__name__)


# ==========================================================
# LOAD MODEL
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "hidden_gem_detection.pkl"
)

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
scaler = bundle["scaler"]
pca = bundle["pca"]
encoders = bundle["label_encoders"]
feature_columns = bundle["feature_columns"]


# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================================
# PREDICTION
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    player = {}

    # Numerical Features

    player["matches_played"] = float(request.form["matches_played"])
    player["total_minutes"] = float(request.form["total_minutes"])
    player["total_goals"] = float(request.form["total_goals"])
    player["total_assists"] = float(request.form["total_assists"])

    player["goals_per_match"] = float(request.form["goals_per_match"])
    player["assists_per_match"] = float(request.form["assists_per_match"])
    player["minutes_per_match"] = float(request.form["minutes_per_match"])

    player["goals_per_90"] = float(request.form["goals_per_90"])
    player["assists_per_90"] = float(request.form["assists_per_90"])
    player["goal_contributions_per_90"] = float(request.form["goal_contributions_per_90"])

    player["estimated_xg"] = float(request.form["estimated_xg"])
    player["estimated_xa"] = float(request.form["estimated_xa"])

    player["age"] = float(request.form["age"])
    player["height_in_cm"] = float(request.form["height_in_cm"])

    player["preferred_foot"] = request.form["preferred_foot"]
    player["position_group"] = request.form["position_group"]

    player["goals_score"] = float(request.form["goals_score"])
    player["assists_score"] = float(request.form["assists_score"])
    player["experience_score"] = float(request.form["experience_score"])

    player["xg_score"] = float(request.form["xg_score"])
    player["xa_score"] = float(request.form["xa_score"])

    player["performance_score"] = float(request.form["performance_score"])
    player["physicality_score"] = float(request.form["physicality_score"])

    player["talent_score"] = float(request.form["talent_score"])



    player = pd.DataFrame([player])



    # ------------------------------------------------------
    # Encode
    # ------------------------------------------------------

    for column in encoders:

        player[column] = encoders[column].transform(
            player[column]
        )



    # ------------------------------------------------------
    # Keep only model features
    # ------------------------------------------------------

    player = player[feature_columns]



    # ------------------------------------------------------
    # Scale
    # ------------------------------------------------------

    player_scaled = scaler.transform(player)



    # ------------------------------------------------------
    # PCA
    # ------------------------------------------------------

    player_pca = pca.transform(player_scaled)



    # ------------------------------------------------------
    # Predict
    # ------------------------------------------------------

    prediction = model.predict(player_pca)[0]

    anomaly_score = model.score_samples(player_pca)[0]



    if prediction == -1:

        result = "Hidden Gem"

    else:

        result = "Regular Player"



    return render_template(

        "index.html",

        prediction=result,

        score=round(anomaly_score, 4)

    )


# ==========================================================
# RUN APP
# ==========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )