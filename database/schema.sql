-- ==========================================================
-- RAW PLAYER DATA
-- ==========================================================

-- ==========================================================
-- RECOMMENDATION HISTORY
-- ==========================================================

CREATE TABLE IF NOT EXISTS recommendation_history (

    id SERIAL PRIMARY KEY,

    player_name VARCHAR(100),

    recommended_player VARCHAR(100),

    similarity_score FLOAT,

    recommendation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP

);


-- ==========================================================
-- HIDDEN GEM PREDICTIONS
-- ==========================================================

CREATE TABLE IF NOT EXISTS hidden_gem_prediction (

    id SERIAL PRIMARY KEY,

    prediction VARCHAR(50),

    anomaly_score FLOAT,

    hidden_gem BOOLEAN,

    potential VARCHAR(100),

    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP

);


-- ==========================================================
-- SAVED PLAYERS
-- ==========================================================

CREATE TABLE IF NOT EXISTS saved_players (

    id SERIAL PRIMARY KEY,

    player_name VARCHAR(100),

    source_module VARCHAR(50),

    saved_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP

);