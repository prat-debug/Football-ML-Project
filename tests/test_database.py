from database.database_utils import save_recommendation

print("Testing database insert...")

save_recommendation(

    player_name="Lionel Messi",

    recommended_player="Neymar",

    similarity_score=97.45

)

print("Success!")