from sqlalchemy import text

from database.db import get_connection


# ==========================================================
# Save Recommendation History
# ==========================================================

def save_recommendation(
    player_name,
    recommended_player,
    recommendation_score,
    method
):

    query = text("""
        INSERT INTO recommendation_history
        (
            player_name,
            recommended_player,
            recommendation_score,
            method
        )
        VALUES
        (
            :player_name,
            :recommended_player,
            :recommendation_score,
            :method
        )
    """)

    with get_connection() as conn:

        conn.execute(
            query,
            {
                "player_name": player_name,
                "recommended_player": recommended_player,
                "recommendation_score": recommendation_score,
                "method": method
            }
        )

        conn.commit()


# ==========================================================
# Save Hidden Gem Prediction
# ==========================================================

def save_hidden_gem(
    prediction,
    anomaly_score,
    hidden_gem,
    potential
):

    query = text("""
        INSERT INTO hidden_gem_prediction
        (
            prediction,
            anomaly_score,
            hidden_gem,
            potential
        )
        VALUES
        (
            :prediction,
            :anomaly_score,
            :hidden_gem,
            :potential
        )
    """)

    with get_connection() as conn:

        conn.execute(
            query,
            {
                "prediction": prediction,
                "anomaly_score": anomaly_score,
                "hidden_gem": hidden_gem,
                "potential": potential
            }
        )

        conn.commit()


# ==========================================================
# Save Player
# ==========================================================

def save_player(
    player_name,
    source_module
):

    query = text("""
        INSERT INTO saved_players
        (
            player_name,
            source_module
        )
        VALUES
        (
            :player_name,
            :source_module
        )
    """)

    with get_connection() as conn:

        conn.execute(
            query,
            {
                "player_name": player_name,
                "source_module": source_module
            }
        )

        conn.commit()

# ==========================================================
# Get Saved Players
# ==========================================================

def get_saved_players():

    query = text("""
        SELECT
            id,
            player_name,
            source_module,
            saved_time
        FROM saved_players
        ORDER BY saved_time DESC
    """)

    with get_connection() as conn:

        result = conn.execute(query)

        return result.mappings().all()