# ==========================================================
# IMPORTS
# ==========================================================
import sys

from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask import session

from src.recommendation import similarity_utils

sys.modules["similarity_utils"] = similarity_utils

from src.recommendation.similarity_utils import load_engine

# ==========================================================
# HIDDEN FEATURES
# ==========================================================

from src.hidden_gem.hidden_gem_utils import (
    load_hidden_gem_model,
    predict_hidden_gem
)

# ==========================================================
# CLUSTERING IMPORTS
# ==========================================================

from src.clustering import predict_cluster

sys.modules["predict_cluster"] = predict_cluster

from src.clustering.predict_cluster import load_pipeline
from src.clustering.predict_cluster import predict_player

# ==========================================================
# DATABASE IMPORTS
# ==========================================================
from database.database_utils import save_recommendation
from database.database_utils import save_hidden_gem
from database.database_utils import save_player
from database.database_utils import get_saved_players

# ==========================================================
# CHATBOT IMPORTS
# ==========================================================

from uuid import uuid4
from chatbot.engine import ChatEngine

print("\nLoading Football Scout AI...")

chat_sessions = {}

print("Football Scout AI Ready.\n")

# ==========================================================
# Football Scout AI
# ==========================================================

def get_chat_engine():

    if "chat_session_id" not in session:

        session["chat_session_id"] = str(
            uuid4()
        )

    session_id = session[
        "chat_session_id"
    ]

    if session_id not in chat_sessions:

        chat_sessions[
            session_id
        ] = ChatEngine()

    return chat_sessions[
        session_id
    ]

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
# LOAD HIDDEN DETECTION MODEL
# ==========================================================

print("Loading Hidden Gem Detection Model...")
hidden_gem_pipeline = load_hidden_gem_model()

print("Hidden Gem Model Loaded.\n")

# ==========================================================
# CREATE FLASK APP
# ==========================================================

app = Flask(__name__)
app.secret_key = "football_talent_discovery_secret_key"


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
        print(recommendations.columns.tolist())
        print(recommendations.head())
        # Save recommendation history
        # we save only the best recommendation.

        if not recommendations.empty:

            first_player = recommendations.iloc[0]

            if method == "knn":

                recommendation_score = float(
                    first_player["Distance"]
                )

            else:

                recommendation_score = float(
                    first_player["Similarity"]
                )

            save_recommendation(
                player_name=player_name,

                recommended_player=first_player["Player"],

                recommendation_score=recommendation_score,

                method=method
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
# SAVE RECOMMENDED PLAYERS
# ==========================================================

@app.route("/save-player", methods=["POST"])
def save_player_route():

    data = request.get_json()

    player_name = data.get("player_name")

    source_module = data.get("source_module")

    try:

        save_player(

            player_name=player_name,

            source_module=source_module

        )

        return jsonify({

            "success": True

        })

    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500

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
# PREDICT HIDDEN GEM
# ==========================================================

@app.route("/hidden-gem", methods=["GET", "POST"])
def hidden_gem():

    result = None
    error = None

    if request.method == "POST":

        try:

            player_data = {

                "matches_played":
                    float(request.form["matches_played"]),

                "total_minutes":
                    float(request.form["total_minutes"]),

                "total_goals":
                    float(request.form["total_goals"]),

                "total_assists":
                    float(request.form["total_assists"]),

                "goals_per_match":
                    float(request.form["goals_per_match"]),

                "assists_per_match":
                    float(request.form["assists_per_match"]),

                "minutes_per_match":
                    float(request.form["minutes_per_match"]),

                "goals_per_90":
                    float(request.form["goals_per_90"]),

                "assists_per_90":
                    float(request.form["assists_per_90"]),

                "goal_contributions_per_90":
                    float(request.form["goal_contributions_per_90"]),

                "estimated_xg":
                    float(request.form["estimated_xg"]),

                "estimated_xa":
                    float(request.form["estimated_xa"]),

                "age":
                    float(request.form["age"]),

                "height_in_cm":
                    float(request.form["height_in_cm"]),

                "preferred_foot":
                    request.form["preferred_foot"],

                "position_group":
                    request.form["position_group"],

                "goals_score":
                    float(request.form["goals_score"]),

                "assists_score":
                    float(request.form["assists_score"]),

                "experience_score":
                    float(request.form["experience_score"]),

                "xg_score":
                    float(request.form["xg_score"]),

                "xa_score":
                    float(request.form["xa_score"]),

                "performance_score":
                    float(request.form["performance_score"]),

                "physicality_score":
                    float(request.form["physicality_score"]),

                "talent_score":
                    float(request.form["talent_score"])

            }

            result = predict_hidden_gem(
                player_data,
                hidden_gem_pipeline
            )
            print(result)
            # Save the result
            save_hidden_gem(
                prediction=result["prediction"],

                anomaly_score=float(result["score"]),

                hidden_gem=bool(result["hidden_gem"]),

                potential=result["interpretation"]
            )

        except Exception as e:

            error = str(e)

    return render_template(

        "hidden_gem.html",

        result=result,

        error=error

    )

# ==========================================================
# GET SAVED PLAYERS
# ==========================================================

@app.route("/saved-players")
def saved_players():

    players = get_saved_players()

    return render_template(

        "saved_players.html",

        players=players

    )

# ==========================================================
# CHATBOT
# ==========================================================

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    user_message = data.get("message", "").strip()

    if not user_message:

        return jsonify({

            "success": False,

            "error": "Message cannot be empty."

        }), 400

    try:

        engine = get_chat_engine()

        reply = engine.send(user_message)

        return jsonify({

            "success": True,

            "reply": reply.text

        })

    except Exception as e:

        print(e)

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


@app.route("/chat-test")
def chat_test():

    engine = get_chat_engine()

    reply = engine.send("Who is Erling Haaland?")

    return f"<pre>{reply.text}</pre>"

@app.route("/scout-ai")
def scout_ai():
    return render_template("scout_ai.html")

@app.route("/ask-scout-ai", methods=["POST"])
def ask_scout_ai():

    try:

        data = request.get_json()

        question = data.get("question", "").strip()

        if not question:

            return jsonify({

                "success": False,

                "error": "Please enter a question."

            }), 400

        engine = get_chat_engine()

        reply = engine.send(question)

        return jsonify({

            "success": True,

            "answer": reply.text

        })

    except Exception as e:

        print(e)

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500

# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    app.run(

        debug=True

    )