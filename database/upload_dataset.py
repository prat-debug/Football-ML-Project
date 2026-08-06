import pandas as pd
from database.db import engine

# ----------------------------------------------------
# Load CSV
# ----------------------------------------------------

EXCEL_PATH = "data/raw/final_destination.xlsx"
df = pd.read_excel(EXCEL_PATH)
print(f"Loaded {len(df)} players.")

# ----------------------------------------------------
# Upload to Neon
# ----------------------------------------------------

df.to_sql(
    name="raw_player_data",
    con=engine,
    if_exists="replace",
    index=False
)

print("Raw dataset uploaded successfully!")