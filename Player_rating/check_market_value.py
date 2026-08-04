"""Evidence that latest_market_value is joined to the wrong player rows.

Run this before trusting any result that uses market value as a target or as a
validation signal.

Usage:
    source myenv/bin/activate
    python check_market_value.py
"""

import pandas as pd

from common import load

PERF = [
    "goal_contributions_per_90",
    "goals_per_90",
    "assists_per_90",
    "minutes_per_match",
    "matches_played",
    "total_goals",
    "age",
    "talent_score",
]

ELITE = [
    "Lionel Messi",
    "Erling Haaland",
    "Jude Bellingham",
    "Vinicius Junior",
    "Kevin De Bruyne",
]


def main():
    pd.set_option("display.width", 200)
    df = load()
    nz = df[df["latest_market_value"] > 0]

    print(f"non-zero market values: {len(nz)} / {len(df)}  ({100 * len(nz) / len(df):.1f}%)\n")

    print("TEST 1 -- correlation with performance (expect strong, got ~0)")
    for c in PERF:
        print(
            f"  {c:<28} pearson={nz[c].corr(nz['latest_market_value']):+.3f}"
            f"  spearman={nz[c].corr(nz['latest_market_value'], method='spearman'):+.3f}"
        )
    print(
        f"  {'highest_market_value':<28} "
        f"pearson={nz['highest_market_value'].corr(nz['latest_market_value']):+.3f}"
        "   <-- only the other value column agrees\n"
    )

    print("TEST 2 -- who are the most valuable players? (expect superstars)")
    print(
        df.nlargest(8, "latest_market_value")[
            ["player_name", "age", "position_group", "matches_played", "total_goals",
             "latest_market_value"]
        ].to_string(index=False)
    )

    print("\nTEST 3 -- what value did actual superstars get? (expect very high)")
    for name in ELITE:
        row = df[df["player_name"] == name]
        if len(row):
            print(
                f"  {name:<20} value={row['latest_market_value'].iloc[0]:>12,.0f}"
                f"  goals={row['total_goals'].iloc[0]:>4}"
            )

    print(
        "\nVERDICT: the market-value block is internally consistent (0.85 with"
        "\nhighest_market_value) but uncorrelated with performance. It is attached"
        "\nto the wrong rows. Fix the upstream join before using it."
    )


if __name__ == "__main__":
    main()
