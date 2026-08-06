import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

# Location of this Python script
# Expected location:
# football-talent-discovery-ml/src/preprocessing.py
BASE_DIR = Path(__file__).resolve().parent

# Project root directory
# Expected location:
# football-talent-discovery-ml/
PROJECT_ROOT = BASE_DIR.parent


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

# Raw data directory
RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

# Cleaned data directory
CLEANED_DIR = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
)

# Results directory
RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)


# Create directories automatically if they do not exist
CLEANED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT AND OUTPUT FILES
# ============================================================

# # Raw input file
# INPUT_FILE = (
#     RAW_DIR
#     / "final_destination.xlsx"
# )

# # Clean master dataset
# CLEAN_FILE = (
#     CLEANED_DIR
#     / "clean_player_features.csv"
# )

# # Preprocessing report
# REPORT_FILE = (
#     RESULTS_DIR
#     / "preprocessing_report.csv"
# )


# # ============================================================
# # 1. LOAD DATA
# # ============================================================

# print("=" * 80)
# print("FOOTBALL TALENT DISCOVERY - COMMON PREPROCESSING PIPELINE")
# print("=" * 80)

# print("\nLoading dataset...")

# print("\nProject Root:")
# print(PROJECT_ROOT)

# print("\nRaw Input File:")
# print(INPUT_FILE)

# print("\nCleaned Dataset Output:")
# print(CLEAN_FILE)

# print("\nPreprocessing Report Output:")
# print(REPORT_FILE)


# if not INPUT_FILE.exists():

#     print("\nERROR: Input file not found!")

#     print("\nPython is looking for the file here:")
#     print(INPUT_FILE)

#     print(
#         "\nPlease place 'final_destination.xlsx' "
#         "inside the following directory:"
#     )

#     print(
#         RAW_DIR
#     )

#     raise FileNotFoundError(
#         f"File not found: {INPUT_FILE}"
#     )


# df = pd.read_excel(
#     INPUT_FILE
# )


# # Store original shape before any preprocessing
# original_shape = df.shape


# print("\nDataset loaded successfully.")

# print(
#     "Original Shape:",
#     original_shape
# )

from database.db import load_raw_player_data

print("=" * 80)
print("FOOTBALL TALENT DISCOVERY - COMMON PREPROCESSING PIPELINE")
print("=" * 80)

print("\nLoading dataset from Neon PostgreSQL...")

df = load_raw_player_data()

original_shape = df.shape

print("\nDataset loaded successfully from Neon.")

print("Original Shape:", original_shape)

# ============================================================
# 2. STANDARDIZE COLUMN NAMES
# ============================================================

print("\nStandardizing column names...")

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)


print("\nColumns detected:")

for column in df.columns:

    print(
        " -",
        column
    )


# ============================================================
# 3. REMOVE UNIVERSALLY IRRELEVANT COLUMNS
# ============================================================

print("\n" + "=" * 80)
print("REMOVING UNIVERSALLY IRRELEVANT COLUMNS")
print("=" * 80)


# These columns have been identified as not being required
# for any of the five ML problems.
#
# Only columns that are definitely irrelevant to ALL problems
# should be placed here.
#
# Columns that may be useful for at least one problem are kept.
#
# A column that is a target (Y) in one problem and a feature (X)
# in another problem is also kept.
#
# Problem-specific feature selection is performed by each
# individual ML pipeline.


UNIVERSALLY_IRRELEVANT_COLUMNS = [

    # Add only columns confirmed to be irrelevant
    # to all five problems here.
]


columns_to_remove = [

    column

    for column in UNIVERSALLY_IRRELEVANT_COLUMNS

    if column in df.columns

]


if columns_to_remove:

    print(
        "Columns removed:"
    )

    for column in columns_to_remove:

        print(
            " -",
            column
        )


    df = df.drop(
        columns=columns_to_remove
    )


else:

    print(
        "No universally irrelevant columns "
        "were found for removal."
    )


# ============================================================
# 4. CREATE PREPROCESSING REPORT
# ============================================================

report = []


def add_report(
    step,
    column,
    action,
    before,
    after,
    notes=""
):

    report.append({

        "step": step,

        "column": column,

        "action": action,

        "before": before,

        "after": after,

        "notes": notes

    })


# ============================================================
# 5. EXACT DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 80)
print("CHECKING EXACT DUPLICATE ROWS")
print("=" * 80)


duplicate_rows = df.duplicated().sum()


print(
    "Exact duplicate rows found:",
    duplicate_rows
)


if duplicate_rows > 0:

    df = df.drop_duplicates()

    print(
        "Removed exact duplicate rows."
    )

else:

    print(
        "No exact duplicate rows found."
    )


add_report(

    "Duplicate Handling",

    "All Columns",

    "Remove exact duplicate rows",

    duplicate_rows,

    0,

    "Exact duplicate records removed."

)


# ============================================================
# 6. PLAYER KEY VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("PLAYER KEY VALIDATION")
print("=" * 80)


if "player_key" in df.columns:

    null_player_keys = (
        df["player_key"].isna().sum()
    )

    duplicate_player_keys = (
        df["player_key"].duplicated().sum()
    )


    print(
        "Null player_key:",
        null_player_keys
    )

    print(
        "Duplicate player_key:",
        duplicate_player_keys
    )


    # player_key is the unique player identifier.
    # Missing player IDs cannot be safely imputed.
    # Therefore, rows with missing player_key are removed.


    if null_player_keys > 0:

        df = df[
            df["player_key"].notna()
        ]


        print(
            f"Removed {null_player_keys} "
            "records with missing player_key."
        )


        add_report(

            "Player Identity",

            "player_key",

            "Remove rows with missing player_key",

            null_player_keys,

            0,

            "Player key is required as unique player identifier."

        )


# ============================================================
# 7. REMOVE INFINITE VALUES
# ============================================================

print("\n" + "=" * 80)
print("CHECKING INFINITE VALUES")
print("=" * 80)


numeric_columns = df.select_dtypes(
    include=np.number
).columns


infinite_report = {}


for column in numeric_columns:

    infinite_count = np.isinf(
        df[column]
    ).sum()

    infinite_report[column] = (
        infinite_count
    )


total_infinite = sum(
    infinite_report.values()
)


print(
    "Total infinite values:",
    total_infinite
)


if total_infinite > 0:

    for column, count in infinite_report.items():

        if count > 0:

            print(
                f"{column}: {count}"
            )


            # Replace infinity with NaN.
            # NaN values are handled in later preprocessing steps.

            df[column] = df[column].replace(

                [np.inf, -np.inf],

                np.nan

            )


# ============================================================
# 8. CATEGORICAL COLUMNS
# ============================================================

print("\n" + "=" * 80)
print("HANDLING CATEGORICAL COLUMNS")
print("=" * 80)


categorical_columns = [

    "player_name",

    "first_name",

    "last_name",

    "country",

    "sub_position",

    "preferred_foot",

    "current_club_name",

    "position_group"

]


for column in categorical_columns:

    if column not in df.columns:

        continue


    missing_before = (
        df[column].isna().sum()
    )


    if missing_before > 0:

        # Missing categorical values are not replaced
        # with the mode because that could introduce
        # incorrect football information.
        #
        # Example:
        # Missing position should not automatically become
        # "Central Midfield".
        #
        # "Unknown" preserves the fact that the information
        # was unavailable.

        df[column] = df[column].fillna(
            "Unknown"
        )


    missing_after = (
        df[column].isna().sum()
    )


    print(

        f"{column}: "
        f"{missing_before} missing -> "
        f"{missing_after} missing"

    )


    add_report(

        "Categorical Missing Values",

        column,

        "Replace NULL with 'Unknown'",

        missing_before,

        missing_after,

        "Missing categorical information preserved as Unknown."

    )


# ============================================================
# 9. NUMERICAL COLUMNS
# ============================================================

print("\n" + "=" * 80)
print("HANDLING NUMERICAL MISSING VALUES")
print("=" * 80)


# ------------------------------------------------------------
# 9A. Numerical columns where zero has football meaning
# ------------------------------------------------------------

zero_meaning_columns = [

    "matches_played",

    "total_minutes",

    "total_goals",

    "total_assists",

    "total_goal_contributions",

    "transfer_count",

    "latest_transfer_fee",

    "highest_transfer_fee"

]


for column in zero_meaning_columns:

    if column not in df.columns:

        continue


    missing_before = (
        df[column].isna().sum()
    )


    if missing_before > 0:

        # For these aggregated football activity and transfer
        # statistics, zero can represent no recorded activity.
        #
        # Therefore, NULL is converted to zero based on the
        # interpretation of these aggregated fields.
        #
        # This does NOT mean that every NULL numerical value
        # in the dataset should be replaced with zero.

        df[column] = df[column].fillna(
            0
        )


    missing_after = (
        df[column].isna().sum()
    )


    print(

        f"{column}: "
        f"{missing_before} missing -> "
        f"{missing_after} missing"

    )


    add_report(

        "Numerical Missing Values",

        column,

        "Replace NULL with 0",

        missing_before,

        missing_after,

        "Zero is meaningful for activity/transfer statistics."

    )


# ============================================================
# 10. MARKET VALUE COLUMNS
# ============================================================

print("\n" + "=" * 80)
print("MARKET VALUE HANDLING")
print("=" * 80)


market_value_columns = [

    "latest_market_value",

    "highest_market_value"

]


for column in market_value_columns:

    if column not in df.columns:

        continue


    missing_before = (
        df[column].isna().sum()
    )


    # Market value columns are retained because they may be
    # required by the market-value problem or other ML tasks.
    #
    # We do NOT create zero_flag columns.
    #
    # The common preprocessing pipeline treats NULL market
    # values as zero because the source dataset already uses
    # zero to represent unavailable market value information.
    #
    # Individual ML pipelines can apply additional
    # problem-specific handling if required.


    df[column] = df[column].fillna(
        0
    )


    missing_after = (
        df[column].isna().sum()
    )


    print(

        f"{column}: "
        f"{missing_before} missing -> "
        f"{missing_after} missing"

    )


    add_report(

        "Market Value Handling",

        column,

        "Fill NULL with 0",

        missing_before,

        missing_after,

        "Market value retained for ML problems that require it. No zero flag created."

    )


# ============================================================
# 11. OTHER NUMERICAL COLUMNS
# ============================================================

print("\n" + "=" * 80)
print("HANDLING OTHER NUMERICAL NULL VALUES")
print("=" * 80)


# These columns should not automatically be interpreted
# as zero when information is unavailable.
#
# Examples:
#
# height_in_cm = NULL
# age = NULL
#
# Median imputation is used for the common master dataset.
#
# A missing flag is created to preserve information about
# the original missing value.


special_numerical_columns = [

    "height_in_cm",

    "age"

]


for column in special_numerical_columns:

    if column not in df.columns:

        continue


    missing_before = (
        df[column].isna().sum()
    )


    if missing_before > 0:

        missing_flag = (

            column +

            "_missing_flag"

        )


        # Preserve information that the value was originally missing.

        df[missing_flag] = (

            df[column]

            .isna()

            .astype(int)

        )


        median_value = (

            df[column]

            .median()

        )


        df[column] = (

            df[column]

            .fillna(

                median_value

            )

        )


        print(

            f"{column}: "
            f"{missing_before} missing values "
            f"filled with median "
            f"{median_value}"

        )


        add_report(

            "Numerical Missing Values",

            column,

            "Median imputation + missing flag",

            missing_before,

            0,

            "Median used because zero is not a valid football value."

        )


# ============================================================
# 12. REMAINING NUMERICAL NULL VALUES
# ============================================================

print("\n" + "=" * 80)
print("CHECKING REMAINING NUMERICAL NULL VALUES")
print("=" * 80)


numeric_columns = df.select_dtypes(
    include=np.number
).columns


for column in numeric_columns:

    missing_before = (
        df[column].isna().sum()
    )


    if missing_before == 0:

        continue


    # For numerical columns not explicitly defined as
    # zero-based, median imputation is used.
    #
    # A missing indicator is also created to preserve
    # information about the original missing value.


    missing_flag = (

        column +

        "_missing_flag"

    )


    df[missing_flag] = (

        df[column]

        .isna()

        .astype(int)

    )


    median_value = (

        df[column]

        .median()

    )


    df[column] = (

        df[column]

        .fillna(

            median_value

        )

    )


    print(

        f"{column}: "
        f"{missing_before} missing -> "
        f"median {median_value}"

    )


    add_report(

        "Remaining Numerical Missing Values",

        column,

        "Median imputation + missing flag",

        missing_before,

        0,

        "Used only for numerical columns not explicitly defined as zero-based."

    )


# ============================================================
# 13. NEGATIVE VALUE VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("CHECKING IMPOSSIBLE NEGATIVE VALUES")
print("=" * 80)


non_negative_columns = [

    "matches_played",

    "total_minutes",

    "total_goals",

    "total_assists",

    "total_goal_contributions",

    "height_in_cm",

    "age",

    "transfer_count",

    "latest_market_value",

    "highest_market_value",

    "latest_transfer_fee",

    "highest_transfer_fee"

]


for column in non_negative_columns:

    if column not in df.columns:

        continue


    negative_count = (

        df[column] < 0

    ).sum()


    print(

        f"{column}: "
        f"{negative_count} negative values"

    )


    if negative_count > 0:

        # Negative values are logically impossible for
        # these football metrics.
        #
        # Replace invalid negative values with NaN,
        # then use median imputation.


        df.loc[

            df[column] < 0,

            column

        ] = np.nan


        median_value = (

            df[column]

            .median()

        )


        df[column] = (

            df[column]

            .fillna(

                median_value

            )

        )


        add_report(

            "Negative Value Handling",

            column,

            "Replace impossible negative values with median",

            negative_count,

            0,

            "Negative values are invalid for this football metric."

        )


# ============================================================
# 14. ENGINEERED FEATURE VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("VALIDATING ENGINEERED FEATURES")
print("=" * 80)


engineered_columns = [

    "goals_per_match",

    "assists_per_match",

    "minutes_per_match",

    "goals_per_90",

    "assists_per_90",

    "goal_contributions_per_90",

    "estimated_xg",

    "estimated_xa",

    "market_value_efficiency",

    "goals_score",

    "assists_score",

    "goal_contribution_score",

    "experience_score",

    "minutes_score",

    "height_score",

    "xg_score",

    "xa_score",

    "value_efficiency_score",

    "performance_score",

    "physicality_score",

    "talent_score",

    "talent_rank"

]


for column in engineered_columns:

    if column not in df.columns:

        continue


    null_count = (
        df[column].isna().sum()
    )


    infinite_count = np.isinf(
        df[column]
    ).sum()


    print(

        f"{column}: "
        f"NULL = {null_count}, "
        f"INF = {infinite_count}"

    )


# ============================================================
# 15. FINAL DATASET VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("FINAL DATASET VALIDATION")
print("=" * 80)


# Store final dataset shape after preprocessing
final_shape = df.shape


print(

    "\nOriginal Dataset Shape:",

    original_shape

)


print(

    "Final Dataset Shape:",

    final_shape

)


print(

    "\nRemaining NULL Values:",

    df.isna().sum().sum()

)


print(

    "Remaining Infinite Values:",

    np.isinf(

        df.select_dtypes(

            include=np.number

        )

    ).sum().sum()

)


print(

    "Duplicate Rows:",

    df.duplicated().sum()

)


if "player_key" in df.columns:

    print(

        "Duplicate Player Keys:",

        df["player_key"].duplicated().sum()

    )


# ============================================================
# 16. FINAL NULL REPORT
# ============================================================

print("\n" + "=" * 80)
print("FINAL NULL REPORT")
print("=" * 80)


final_missing_report = (

    df.isna()

    .sum()

    .sort_values(

        ascending=False

    )

)


print(

    final_missing_report[

        final_missing_report > 0

    ]

)


# ============================================================
# 17. SAVE CLEAN DATASET
# ============================================================

print("\n" + "=" * 80)
print("SAVING CLEAN DATASET")
print("=" * 80)


df.to_csv(

    CLEAN_FILE,

    index=False

)


print(

    "\nClean dataset saved successfully!"

)


print(

    "Location:",

    CLEAN_FILE

)


# ============================================================
# 18. SAVE PREPROCESSING REPORT
# ============================================================

report_df = pd.DataFrame(
    report
)


report_df.to_csv(

    REPORT_FILE,

    index=False

)


print(

    "\nPreprocessing report saved successfully!"

)


print(

    "Location:",

    REPORT_FILE

)


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 80)


print(

    "\nOriginal Shape:",

    original_shape

)


print(

    "Final Shape:",

    final_shape

)


print(

    "\nOutput Files:"

)


print(

    "1. Clean Dataset:",

    CLEAN_FILE

)


print(

    "2. Preprocessing Report:",

    REPORT_FILE

)


print(

    "\nThe clean master dataset is now ready "

    "for individual ML pipelines."

)


print(

    "\nEach team member should perform "

    "their own feature selection and "

    "model-specific preprocessing "

    "after loading this dataset."

)


print("\nDone.")