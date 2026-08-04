# ==========================================================
# IMPORTS
# ==========================================================

from similarity_utils import load_engine

# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 80)
    print("SIMILAR PLAYER RECOMMENDATION")
    print("=" * 80)

    print("\nLoading trained recommendation engine...")

    engine = load_engine()

    print("Recommendation engine loaded successfully.")

    while True:

        print("\n")

        player_name = input(
            "Enter Player Name (or 'exit'): "
        ).strip()

        if player_name.lower() == "exit":
            break

        print("\nRecommendation Methods")

        print("1. KNN")
        print("2. Cosine Similarity")

        choice = input(
            "\nChoose Method (1-2): "
        ).strip()

        method_map = {

            "1": "knn",

            "2": "cosine"

        }

        if choice not in method_map:

            print("\nInvalid Choice!")

            continue

        method = method_map[choice]

        top_n = input(
            "\nNumber of Recommendations (Default = 10): "
        ).strip()

        if top_n == "":

            top_n = 10

        else:

            try:

                top_n = int(top_n)

            except ValueError:

                print("\nPlease enter a valid number.")

                continue

        try:

            recommendations = engine.recommend(

                player_name=player_name,

                method=method,

                top_n=top_n

            )
            print("\n")
            print("=" * 80)
            print(
                f"Top {top_n} Similar Players "
                f"({method.upper()})"
            )
            print("=" * 80)

            print(
                recommendations.to_string(index=False)
            )

        except ValueError as e:

            print("\nError:")

            print(e)

        except Exception as e:

            print("\nUnexpected Error:")

            print(e)

    print("\nExiting Recommendation System...")

# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()