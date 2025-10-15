import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle

# Load datasets
movies_df = pd.read_csv("data/rotten_tomatoes_movies.csv")
reviews_df = pd.read_csv("data/rotten_tomatoes_movie_reviews.csv")

# Merge datasets on 'id' to incorporate reviews
merged_df = movies_df.merge(reviews_df[['id', 'reviewText']], on='id', how='left')

# Fill missing values before concatenation
merged_df.fillna({'genre': '', 'director': '', 'originalLanguage': '', 'reviewText': '', 
                  'audienceScore': 0, 'tomatoMeter': 0}, inplace=True)

# Combine relevant features
merged_df['features'] = (
    merged_df['genre'] + " " + 
    merged_df['director'] + " " + 
    merged_df['originalLanguage'] + " " + 
    merged_df['reviewText'] + " " + 
    merged_df['audienceScore'].astype(str) + " " + 
    merged_df['tomatoMeter'].astype(str)
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(merged_df['features'])

# Fit Nearest Neighbors model (memory-efficient alternative to cosine similarity)
knn = NearestNeighbors(metric="cosine", algorithm="brute")
knn.fit(tfidf_matrix)

# Save processed dataset
merged_df.to_csv("data/processed_movies.csv", index=False)

# Save models & data
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("models/tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)

with open("models/knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)

# Save movie indices for lookup
movie_indices = pd.Series(merged_df.index, index=merged_df['title']).to_dict()
with open("models/movie_indices.pkl", "wb") as f:
    pickle.dump(movie_indices, f)

print("Preprocessing complete. Model and dataset saved successfully!")
