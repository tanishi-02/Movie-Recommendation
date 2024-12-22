import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "DATA.csv"  # File name set to DATA.csv
movies_df = pd.read_csv(file_path)

# Feature selection
numeric_features = ["Year", "Runtime (Minutes)", "Rating", "Votes", "Revenue (Millions)", "Metascore"]
categorical_features = ["Genre", "Director"]

# One-hot encode categorical features
encoded_features = pd.get_dummies(movies_df[categorical_features])

# Combine numeric and encoded features
features = pd.concat([movies_df[numeric_features], encoded_features], axis=1)

# Fill missing values with the mean
features.fillna(features.mean(), inplace=True)

# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Build the KNN model
knn_model = NearestNeighbors(n_neighbors=6, metric='cosine')  # Set n_neighbors to 6
knn_model.fit(features_scaled)


# Recommendation function
def recommend_movies(title, movies_df, features_scaled, knn_model):
    # Find the index of the movie
    movie_idx = movies_df[movies_df['Title'].str.contains(title, case=False, na=False)].index
    if len(movie_idx) == 0:
        return []

    movie_idx = movie_idx[0]

    # Find neighbors
    distances, indices = knn_model.kneighbors([features_scaled[movie_idx]])

    # Exclude the input movie and limit recommendations to 5 movies
    recommended_movies = movies_df.iloc[indices[0][1:6]]  # [1:6] excludes input movie and keeps 5 movies
    return recommended_movies[['Title', 'Genre', 'Rating', 'Director']]


# Example usage
movie_title = "Inception"  # Replace with a title from your dataset
recommendations = recommend_movies(movie_title, movies_df, features_scaled, knn_model)

if not recommendations.empty:
    print(recommendations)
else:
    print("No recommendations found.")
