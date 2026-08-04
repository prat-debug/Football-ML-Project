import pandas as pd
import joblib
from pathlib import Path

# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "hidden_gem_detection.pkl"
)

# ==========================================================
# LOAD PIPELINE
# ==========================================================

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
scaler = bundle["scaler"]
pca = bundle["pca"]
encoders = bundle["label_encoders"]
feature_columns = bundle["feature_columns"]

# ==========================================================
# PLAYER TO TEST
# ==========================================================

player = {

    "matches_played": 38,
    "total_minutes": 3100,
    "total_goals": 0,
    "total_assists": 0,

    "goals_per_match": 0 / 38,
    "assists_per_match": 0 / 38,
    "minutes_per_match": 3100 / 38,

    "goals_per_90": 0.23,
    "assists_per_90": 0.32,
    "goal_contributions_per_90": 0.55,

    "estimated_xg": 0.27,
    "estimated_xa": 0.35,

    "age": 19,
    "height_in_cm": 178,

    "preferred_foot": "left",
    "position_group": "Winger",

    "goals_score": 4,
    "assists_score": 8,
    "experience_score": 55,

    "xg_score": 2,
    "xa_score": 8,

    "performance_score": 88,
    "physicality_score": 70,

    "talent_score": 66
}

player = pd.DataFrame([player])

# ==========================================================
# KEEP MODEL FEATURES
# ==========================================================

player = player[feature_columns]

# ==========================================================
# ENCODE CATEGORICAL FEATURES
# ==========================================================

for column in encoders:

    player[column] = encoders[column].transform(player[column])

# ==========================================================
# SCALE FEATURES
# ==========================================================

player_scaled = scaler.transform(player)

# ----------------------------------------------------------
# FIXES THE PCA WARNING
# ----------------------------------------------------------

player_scaled = pd.DataFrame(
    player_scaled,
    columns=feature_columns
)

# ==========================================================
# PCA TRANSFORMATION
# ==========================================================

player_pca = pca.transform(player_scaled)

# ==========================================================
# PREDICTION
# ==========================================================

prediction = model.predict(player_pca)[0]

anomaly_score = model.decision_function(player_pca)[0]

# ==========================================================
# INTERPRETATION
# ==========================================================

if prediction == -1:
    result = "HIDDEN GEM"
else:
    result = "REGULAR PLAYER"

if anomaly_score < -0.15:
    potential = "Very High Hidden Gem Potential"

elif anomaly_score < -0.08:
    potential = "High Hidden Gem Potential"

elif anomaly_score < -0.02:
    potential = "Moderate Hidden Gem Potential"

else:
    potential = "Low Hidden Gem Potential"

# ==========================================================
# OUTPUT
# ==========================================================

print("\n" + "=" * 60)

print("Prediction              :", result)

print("Hidden Gem              :", prediction == -1)

print("Anomaly Score           :", round(float(anomaly_score), 4))

print("Interpretation          :", potential)

print("=" * 60)