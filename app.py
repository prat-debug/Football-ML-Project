# ==========================================================
# IMPORTS
# ==========================================================
import sys

from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify

from src.recommendation import similarity_utils

sys.modules["similarity_utils"] = similarity_utils

from src.recommendation.similarity_utils import load_engine

# ==========================================================
# CLUSTERING IMPORTS
# ==========================================================

from src.clustering import predict_cluster

sys.modules["predict_cluster"] = predict_cluster

from src.clustering.predict_cluster import load_pipeline
from src.clustering.predict_cluster import predict_player


# ==========================================================
# LOAD RECOMMENDATION ENGINE
# ==========================================================

print("\nLoading Recommendation Engine...")

engine = load_engine()

print("Recommendation Engine Loaded Successfully.\n")

# ==========================================================
# LOAD CLUSTERING PIPELINE
# ==========================================================

print("Loading Clustering Pipeline...")

cluster_pipeline = load_pipeline()

print("Clustering Pipeline Loaded Successfully.\n")

# ==========================================================
# CREATE FLASK APP
# ==========================================================

app = Flask(__name__)


# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================================
# RECOMMENDATION PAGE
# ==========================================================

@app.route("/recommendation")
def recommendation():

    return render_template("recommendation.html")

# ==========================================================
# PLAYER CLUSTERING PAGE
# ==========================================================

@app.route("/clustering")
def clustering():

    return render_template("clustering.html")

# ==========================================================
# PREDICT SIMILAR PLAYERS
# ==========================================================

@app.route("/recommend", methods=["POST"])
def recommend():

    try:

        player_name = request.form.get("player_name")

        method = request.form.get("method")

        top_n = request.form.get("top_n")

        if top_n == "":
            top_n = 10
        else:
            top_n = int(top_n)

        recommendations = engine.recommend(

            player_name=player_name,

            method=method,

            top_n=top_n

        )

        return jsonify(

            recommendations.to_dict(
                orient="records"
            )

        )

    except Exception as e:

        return jsonify({

            "error": str(e)

        })

# ==========================================================
# PREDICT PLAYER CLUSTER
# ==========================================================

@app.route("/predict-cluster", methods=["POST"])
def predict_cluster_route():

    try:

        player_data = {

            "age": float(request.form.get("age")),

            "height_in_cm": float(request.form.get("height")),

            "matches_played": float(request.form.get("matches")),

            "minutes_per_match": float(request.form.get("minutes")),

            "goals_per_90": float(request.form.get("goals_per_90")),

            "assists_per_90": float(request.form.get("assists_per_90")),

            "estimated_xg": float(request.form.get("estimated_xg")),

            "estimated_xa": float(request.form.get("estimated_xa")),

            "physicality_score": float(request.form.get("physicality_score"))

        }

        cluster_id, category = predict_player(

            player_data,

            cluster_pipeline

        )

        return jsonify({

            "cluster_id": int(cluster_id),

            "category": category

        })

    except Exception as e:

        return jsonify({

            "error": str(e)

        })

# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    app.run(

        debug=True

    )