import os
import joblib
import pandas as pd


# ------------------------------------------------------------
# Model Path
# ------------------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "models",
    "hidden_gem_detection.pkl"
)


# ------------------------------------------------------------
# Load Pipeline
# ------------------------------------------------------------

def load_hidden_gem_model():

    return joblib.load(MODEL_PATH)


# ------------------------------------------------------------
# Interpretation Logic
# ------------------------------------------------------------

def interpret_score(score):

    if score <= -0.10:
        return "High Hidden Gem Potential"

    elif score <= -0.05:
        return "Moderate Hidden Gem Potential"

    elif score <= 0:
        return "Low Hidden Gem Potential"

    else:
        return "Very Low Hidden Gem Potential"


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

def predict_hidden_gem(player_data, pipeline):

    feature_columns = pipeline["feature_columns"]
    encoders = pipeline["label_encoders"]
    scaler = pipeline["scaler"]
    pca = pipeline["pca"]
    model = pipeline["model"]

    player_df = pd.DataFrame([player_data])

    # --------------------------------------------------------
    # Encode categorical columns
    # --------------------------------------------------------

    for column, encoder in encoders.items():
        # print(f"\n{column}")
        # print("Allowed labels:", encoder.classes_)

        player_df[column] = encoder.transform(
            player_df[column]
        )

    # --------------------------------------------------------
    # Reorder columns
    # --------------------------------------------------------

    player_df = player_df[feature_columns]

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    X_scaled = scaler.transform(player_df)

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    X_pca = pca.transform(X_scaled)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(X_pca)[0]

    score = model.decision_function(X_pca)[0]

    hidden_gem = prediction == -1

    result = "Hidden Gem" if hidden_gem else "Regular Player"

    interpretation = interpret_score(score)

    return {
        "prediction": result,
        "hidden_gem": hidden_gem,
        "score": round(float(score), 4),
        "interpretation": interpretation
    }