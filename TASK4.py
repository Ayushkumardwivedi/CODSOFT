import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

user_item_matrix = pd.DataFrame({
    'Spider Man': [4, 0, 0, 5],
    'Dabangg': [0, 0, 5, 4],
    'Tere Naam': [5, 4, 0, 0],
    'Salaar': [0, 5, 4, 0]
})

item_similarity_matrix = pd.DataFrame(cosine_similarity(user_item_matrix.T), index=user_item_matrix.columns, columns=user_item_matrix.columns)

def get_recommendations(user_id):
    user_ratings = user_item_matrix.iloc[user_id]
    weighted_ratings = user_ratings.dot(item_similarity_matrix)
    recommendations = weighted_ratings.sort_values(ascending=False).index.tolist()
    return recommendations

user_id = 3
recommendations = get_recommendations(user_id)
print(f"Recommendations for User {user_id}: {recommendations}")
