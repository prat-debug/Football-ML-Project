from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Location of this Python script
# Expected location:
# football-talent-discovery-ml/src/profiling.py
BASE_DIR = Path(__file__).resolve().parent

# Project root directory
# Expected location:
# football-talent-discovery-ml/
PROJECT_ROOT = BASE_DIR.parent


# ============================================================
# 2. RAW AND RESULTS DIRECTORIES
# ============================================================

# Raw data directory
RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

# Profiling reports directory
RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "profiling"
)


# Create profiling results directory automatically
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. INPUT FILE
# ============================================================

# Raw Excel dataset
INPUT_FILE = (
    RAW_DIR
    / "final_destination.xlsx"
)


# ============================================================
# 4. DISPLAY PATH INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("PROJECT PATH INFORMATION")
print("=" * 70)

print("Project Root:")
print(PROJECT_ROOT)

print("\nScript Location:")
print(BASE_DIR)

print("\nRaw Excel File:")
print(INPUT_FILE)

print("\nProfiling Results Folder:")
print(RESULTS_DIR)


# ============================================================
# 5. CHECK WHETHER EXCEL FILE EXISTS
# ============================================================

if not INPUT_FILE.exists():

    print(
        "\nERROR: final_destination.xlsx "
        "was not found!"
    )


    print(
        "\nPython is looking for the file here:"
    )


    print(
        INPUT_FILE
    )


    print(
        "\nExpected repository structure:"
    )


    print("""
football-talent-discovery-ml/
│
├── data/
│   ├── raw/
│   │   └── final_destination.xlsx
│   │
│   └── cleaned/
│
├── src/
│   ├── preprocessing.py
│   └── profiling.py
│
└── results/
    └── profiling/
    """)


    raise FileNotFoundError(
        f"Excel file not found at: {INPUT_FILE}"
    )


# ============================================================
# 6. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)


df = pd.read_excel(
    INPUT_FILE
)


print(
    "\nDataset loaded successfully!"
)


print(
    "Dataset Shape:",
    df.shape
)


print(
    "\nColumns:"
)


print(
    df.columns.tolist()
)


# ============================================================
# 7. BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)


df.info()


# ============================================================
# 8. MISSING VALUE ANALYSIS
# ============================================================

missing_report = pd.DataFrame({

    "column": df.columns,

    "missing_count": (

        df.isnull()

        .sum()

        .values

    ),

    "missing_percentage": (

        df.isnull()

        .mean()

        .values

        * 100

    ).round(2)

})


# Sort by number of missing values

missing_report = (

    missing_report

    .sort_values(

        by="missing_count",

        ascending=False

    )

)


print("\n" + "=" * 70)
print("MISSING VALUE REPORT")
print("=" * 70)


# Display only columns containing missing values

missing_columns = missing_report[

    missing_report["missing_count"] > 0

]


if len(missing_columns) > 0:

    print(

        missing_columns

        .to_string(

            index=False

        )

    )

else:

    print(
        "No missing values found."
    )


# ============================================================
# 9. ZERO VALUE ANALYSIS
# ============================================================

zero_report = pd.DataFrame({

    "column": df.columns,

    "zero_count": [

        (

            df[col] == 0

        ).sum()

        if pd.api.types.is_numeric_dtype(

            df[col]

        )

        else np.nan

        for col in df.columns

    ]

})


zero_report["zero_percentage"] = (

    zero_report["zero_count"]

    / len(df)

    * 100

).round(2)


# Sort by zero count

zero_report = (

    zero_report

    .sort_values(

        by="zero_count",

        ascending=False

    )

)


print("\n" + "=" * 70)
print("ZERO VALUE REPORT")
print("=" * 70)


zero_columns = zero_report[

    zero_report["zero_count"] > 0

]


if len(zero_columns) > 0:

    print(

        zero_columns

        .to_string(

            index=False

        )

    )

else:

    print(
        "No zero values found."
    )


# ============================================================
# 10. UNIQUE VALUE ANALYSIS
# ============================================================

unique_report = pd.DataFrame({

    "column": df.columns,

    "unique_values": [

        df[col].nunique(

            dropna=False

        )

        for col in df.columns

    ]

})


print("\n" + "=" * 70)
print("UNIQUE VALUE REPORT")
print("=" * 70)


print(

    unique_report

    .sort_values(

        by="unique_values"

    )

    .to_string(

        index=False

    )

)


# ============================================================
# 11. NUMERICAL STATISTICS
# ============================================================

numeric_columns = (

    df.select_dtypes(

        include=np.number

    )

    .columns

)


numeric_summary = (

    df[numeric_columns]

    .describe()

    .T

)


# Add missing value count

numeric_summary["missing"] = (

    df[numeric_columns]

    .isnull()

    .sum()

)


# Add zero count

numeric_summary["zero_count"] = [

    (

        df[col] == 0

    ).sum()

    for col in numeric_columns

]


# Add zero percentage

numeric_summary["zero_percentage"] = (

    numeric_summary["zero_count"]

    / len(df)

    * 100

).round(2)


print("\n" + "=" * 70)
print("NUMERICAL FEATURE SUMMARY")
print("=" * 70)


print(

    numeric_summary

    .to_string()

)


# ============================================================
# 12. MARKET VALUE INVESTIGATION
# ============================================================

market_columns = [

    "latest_market_value",

    "highest_market_value",

    "latest_transfer_fee",

    "highest_transfer_fee",

    "transfer_count"

]


print("\n" + "=" * 70)
print("MARKET VALUE INVESTIGATION")
print("=" * 70)


# Check which market columns actually exist

available_market_columns = [

    col

    for col in market_columns

    if col in df.columns

]


missing_market_columns = [

    col

    for col in market_columns

    if col not in df.columns

]


if missing_market_columns:

    print(

        "\nWARNING: The following expected columns "
        "were not found:"

    )


    for col in missing_market_columns:

        print(

            "-",

            col

        )


for col in available_market_columns:

    print(
        "\n" + "-" * 50
    )


    print(
        col
    )


    print(

        "Zero Count:",

        (

            df[col] == 0

        ).sum()

    )


    print(

        "Zero Percentage:",

        round(

            (

                df[col] == 0

            ).mean()

            * 100,

            2

        )

    )


    print(

        "Missing Count:",

        df[col].isnull().sum()

    )


    print(

        "Missing Percentage:",

        round(

            df[col].isnull().mean()

            * 100,

            2

        )

    )


# ============================================================
# 13. MARKET VALUE VS TRANSFER HISTORY
# ============================================================

required_columns = [

    "latest_market_value",

    "highest_market_value",

    "transfer_count"

]


if all(

    col in df.columns

    for col in required_columns

):

    print("\n" + "=" * 70)
    print("MARKET VALUE VS TRANSFER HISTORY")
    print("=" * 70)


    market_zero_analysis = df[

        (

            df["latest_market_value"] == 0

        )

        &

        (

            df["highest_market_value"] == 0

        )

    ]


    print(

        "Players with BOTH latest and highest "
        "market value = 0:",

        len(market_zero_analysis)

    )


    print(

        "\nTransfer count distribution "

        "for players with zero market value:"

    )


    print(

        market_zero_analysis[

            "transfer_count"

        ]

        .value_counts()

        .head(20)

    )


else:

    print(

        "\nSkipping Market Value vs Transfer History "

        "analysis because required columns are missing."

    )


# ============================================================
# 14. MARKET VALUE VS PLAYING ACTIVITY
# ============================================================

activity_columns = [

    "latest_market_value",

    "highest_market_value",

    "matches_played",

    "total_minutes",

    "age",

    "transfer_count"

]


if all(

    col in df.columns

    for col in activity_columns

):

    print("\n" + "=" * 70)
    print("MARKET VALUE VS PLAYING ACTIVITY")
    print("=" * 70)


    market_zero_activity = df[

        (

            df["latest_market_value"] == 0

        )

        &

        (

            df["highest_market_value"] == 0

        )

    ]


    print(

        market_zero_activity[

            [

                "matches_played",

                "total_minutes",

                "age",

                "transfer_count"

            ]

        ]

        .describe()

    )


else:

    print(

        "\nSkipping Market Value vs Playing Activity "

        "analysis because required columns are missing."

    )


# ============================================================
# 15. MARKET VALUE VS POSITION
# ============================================================

position_columns = [

    "position_group",

    "player_key",

    "latest_market_value",

    "highest_market_value"

]


if all(

    col in df.columns

    for col in position_columns

):

    print("\n" + "=" * 70)
    print("MARKET VALUE VS POSITION")
    print("=" * 70)


    position_market_analysis = (

        df.groupby(

            "position_group"

        )

        .agg(

            total_players=(

                "player_key",

                "count"

            ),


            zero_latest_market_value=(

                "latest_market_value",

                lambda x:

                (x == 0).sum()

            ),


            zero_highest_market_value=(

                "highest_market_value",

                lambda x:

                (x == 0).sum()

            ),


            median_latest_market_value=(

                "latest_market_value",

                "median"

            ),


            median_highest_market_value=(

                "highest_market_value",

                "median"

            )

        )

    )


    # Calculate zero percentage

    position_market_analysis[

        "zero_latest_percentage"

    ] = (

        position_market_analysis[

            "zero_latest_market_value"

        ]

        /

        position_market_analysis[

            "total_players"

        ]

        *

        100

    ).round(2)


    position_market_analysis[

        "zero_highest_percentage"

    ] = (

        position_market_analysis[

            "zero_highest_market_value"

        ]

        /

        position_market_analysis[

            "total_players"

        ]

        *

        100

    ).round(2)


    print(

        position_market_analysis

        .sort_values(

            "zero_latest_percentage",

            ascending=False

        )

        .to_string()

    )


else:

    print(

        "\nSkipping Market Value vs Position "

        "analysis because required columns are missing."

    )


# ============================================================
# 16. AGE VS MARKET VALUE
# ============================================================

age_columns = [

    "age",

    "latest_market_value"

]


if all(

    col in df.columns

    for col in age_columns

):

    print("\n" + "=" * 70)
    print("AGE VS MARKET VALUE")
    print("=" * 70)


    age_market_analysis = (

        df.groupby(

            "age"

        )

        .agg(

            players=(

                "latest_market_value",

                "count"

            ),


            zero_market_value=(

                "latest_market_value",

                lambda x:

                (x == 0).sum()

            ),


            median_market_value=(

                "latest_market_value",

                "median"

            )

        )

    )


    age_market_analysis[

        "zero_market_value_percentage"

    ] = (

        age_market_analysis[

            "zero_market_value"

        ]

        /

        age_market_analysis[

            "players"

        ]

        *

        100

    ).round(2)


    print(

        age_market_analysis

        .to_string()

    )


else:

    print(

        "\nSkipping Age vs Market Value "

        "analysis because required columns are missing."

    )


# ============================================================
# 17. SAVE REPORTS
# ============================================================

print("\n" + "=" * 70)
print("SAVING REPORTS")
print("=" * 70)


# Save missing value report

missing_report.to_csv(

    RESULTS_DIR /

    "missing_value_report.csv",

    index=False

)


# Save zero value report

zero_report.to_csv(

    RESULTS_DIR /

    "zero_value_report.csv",

    index=False

)


# Save unique value report

unique_report.to_csv(

    RESULTS_DIR /

    "unique_value_report.csv",

    index=False

)


# Save numerical summary

numeric_summary.to_csv(

    RESULTS_DIR /

    "numeric_summary.csv"

)


# Save position analysis if it was created

if "position_market_analysis" in locals():

    position_market_analysis.to_csv(

        RESULTS_DIR /

        "position_market_analysis.csv"

    )


# Save age analysis if it was created

if "age_market_analysis" in locals():

    age_market_analysis.to_csv(

        RESULTS_DIR /

        "age_market_analysis.csv"

    )


# ============================================================
# 18. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("PROFILING COMPLETED SUCCESSFULLY")
print("=" * 70)


print(

    "\nAll available reports have been saved to:"

)


print(
    RESULTS_DIR
)


print(
    "\nGenerated files:"
)


for file in RESULTS_DIR.iterdir():

    if file.is_file():

        print(

            " -",

            file.name

        )


print("\n")