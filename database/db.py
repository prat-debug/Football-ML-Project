import pandas as pd
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://neondb_owner:npg_l1FDOKqbC3UB@ep-floral-cherry-ayx25uve.c-5.us-east-2.aws.neon.tech/neondb?sslmode=require"

engine = create_engine(DATABASE_URL)


def load_raw_player_data():

    query = """
        SELECT *
        FROM raw_player_data
    """

    return pd.read_sql(query, engine)