# item based:baseline
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
user_reviews = pd.read_csv('user_reviews.csv')
movie_genres = pd.read_csv('movie_genres.csv')
header = user_reviews.columns.tolist()[2:]
first_column = user_reviews.iloc[:, 1]
name = first_column.to_numpy()

user_reviews_cleaned = user_reviews.iloc[:, 2:].values  # User-movie ratings
user_reviews_cleaned = user_reviews_cleaned / np.max(user_reviews_cleaned)  # Normalize ratings
movie_genres_cleaned = movie_genres.iloc[:, 2:].values  # Movie-genre matrix


def item_based_recommendation(user_reviews_tensor, num_recommendations=5):
    item_similarity = cosine_similarity(user_reviews_tensor.T)
    recommendations = {}

    # Calculate user averages, ignoring zeros
    user_means = np.array([np.mean(user_reviews_tensor[user_idx][user_reviews_tensor[user_idx] > 0])
                           for user_idx in range(user_reviews_tensor.shape[0])])

    for user_idx in range(user_reviews_tensor.shape[0]):
        user_ratings = user_reviews_tensor[user_idx]

        predicted_ratings = np.zeros_like(user_ratings)
        for movie_idx in range(len(user_ratings)):
            if user_ratings[movie_idx] > 0:
                continue

            weighted_sum = 0
            sim_sum = 0
            for other_movie_idx in range(len(user_ratings)):
                if user_ratings[other_movie_idx] > 0:
                    # Adjust ratings by subtracting the user's average
                    adjusted_rating = user_ratings[other_movie_idx] - user_means[user_idx]
                    weighted_sum += item_similarity[movie_idx, other_movie_idx] * adjusted_rating
                    sim_sum += item_similarity[movie_idx, other_movie_idx]

            # Apply the baseline adjustment
            if sim_sum > 0:
                predicted_ratings[movie_idx] = weighted_sum / sim_sum + user_means[user_idx]

        recommended_movie_indices = np.argsort(-predicted_ratings)[:num_recommendations]
        recommendations[name[user_idx]] = [header[idx] for idx in recommended_movie_indices]

    return recommendations


item_based_rec = item_based_recommendation(user_reviews_cleaned)

print("Item-Based Recommendations:")
for user, movies in item_based_rec.items():
    print(f"{user}: {movies}")

# Recommend top 5 movies for the first 5 users
num_recommendations = 5
for user_id in range(min(5, num_users)):  # Loop through the first 5 users
    user_ratings = R_pred_final[user_id]  # Predicted ratings for this user
    already_rated = user_reviews_cleaned[user_id] > 0  # Movies already rated by the user
    recommendations = np.argsort(-user_ratings)  # Sort movies by predicted rating (descending)

    # Filter out movies already rated
    recommendations = [movie for movie in recommendations if not already_rated[movie]]

    # Get the top 5 recommendations
    top_recommendations = recommendations[:num_recommendations]

    print(f"Recommended movies for User {user_id + 1}: {top_recommendations}")





