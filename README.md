Project Report: Movie Recommendation System Using Content-Based Filtering
1. Introduction
In today’s digital age, movie recommendation systems play a crucial role in enhancing the user experience by suggesting films that align with individual tastes. This project focuses on building a content-based movie recommendation system that leverages movie metadata and critic reviews to suggest similar films. The system uses machine learning techniques to process and analyze movie features, and it is deployed as a web application using FastAPI.

2. Problem Statement
The movie industry offers an overwhelming number of films across various platforms. As a result, viewers often struggle to discover movies that fit their preferences. The goal of this project is to create a system that:

Analyzes multiple movie features such as genres, directors, audience scores, and critic reviews.
Provides personalized recommendations by identifying similar movies based on content.
Enhances user experience by presenting relevant and diverse movie suggestions.
3. Dataset
Two main datasets were used:

Rotten Tomatoes Movies Dataset (rotten_tomatoes_movies.csv):
Contains details like movie title, genre, director, original language, audience score, tomato meter, runtime, and more.
Rotten Tomatoes Movie Reviews Dataset (rotten_tomatoes_movie_reviews.csv):
Includes critic reviews, review text, original score, and related metadata.
These datasets were merged based on the unique movie identifier (id), combining movie metadata with corresponding critic reviews to form a comprehensive view of each film.

4. Methodology
4.1 Data Preprocessing
Merging Data:
The two datasets were merged using the id field. This allowed the system to incorporate both movie metadata (e.g., genre, director) and textual data from critic reviews.

Handling Missing Values and Duplicates:
Missing values in key fields were filled with default values or empty strings. Duplicate movie entries (based on the movie title) were removed to ensure unique recommendations.

Feature Combination:
Relevant features such as genre, director, original language, review text, audience score, and tomato meter were concatenated into a single string for each movie. This combined text representation serves as the input for feature extraction.

4.2 Feature Extraction Using TF-IDF
TF-IDF Vectorization:
The Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer was used to convert the combined textual features into numerical vectors.
Term Frequency (TF): Measures how frequently a term appears in a document.
Inverse Document Frequency (IDF): Weights down the common terms across all documents.
This method produces a matrix where each row represents a movie and each column represents a term from the combined text. The resulting TF-IDF matrix captures the importance of words relative to the overall dataset.
4.3 Building the Recommendation Model with K-Nearest Neighbors (KNN)
KNN Model:
A K-Nearest Neighbors (KNN) model was trained on the TF-IDF matrix. The KNN algorithm calculates the cosine similarity between movie vectors:

Cosine Similarity: Measures the cosine of the angle between two vectors, which indicates their orientation (similarity in content).
The model finds the top k nearest movies (neighbors) that are most similar to the given movie based on their TF-IDF vectors.
Avoiding Memory Issues:
Since the TF-IDF matrix is sparse (mostly zeros), the model uses sparse matrix operations to maintain memory efficiency. Special care was taken to reshape vectors and avoid converting the sparse matrix into a dense format.

4.4 Making Recommendations
When a user inputs a movie title:

Input Normalization: The title is converted to lowercase and stripped of extra spaces.
Fuzzy Matching: If the exact title is not found, fuzzy matching techniques are used to suggest the closest match.
Index Retrieval: The corresponding index for the movie is retrieved from a pre-saved dictionary.
KNN Query: The KNN model finds the nearest neighbors (similar movies) by comparing TF-IDF vectors.
Post-Processing: Recommendations are filtered to remove duplicates and out-of-bound indices.
Display: The final list of recommendations (movie title, genre, director) is rendered on a web page.
5. Implementation
5.1 Technology Stack
Python: The primary programming language.
FastAPI: For building and serving the web application.
Jinja2: For templating and rendering HTML.
Scikit-Learn: For TF-IDF vectorization and KNN modeling.
Pandas & NumPy: For data manipulation and processing.
Pickle: For saving and loading pre-trained models and data structures.
5.2 Project Structure
php
Copy
Edit
movie_recommendation_system/
│
├── data/
│   ├── rotten_tomatoes_movies.csv
│   ├── rotten_tomatoes_movie_reviews.csv
│   └── processed_movies.csv
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_matrix.pkl
│   ├── knn_model.pkl
│   └── movie_indices.pkl
│
├── static/
│   └── bg.jpg          # Background image for UI
│
├── templates/
│   └── index.html      # HTML template with custom styling
│
├── main.py             # FastAPI application code
└── preprocessing.py    # Data preprocessing and model building script
5.3 Code Overview
preprocessing.py:
Reads and merges datasets, cleans the data, performs TF-IDF vectorization, trains the KNN model, and saves the resulting objects.

main.py:
Loads the preprocessed data and pickled models. Handles user requests for movie recommendations using FastAPI routes, processes input with fuzzy matching, and returns recommendations using the KNN model.

index.html:
Provides a user-friendly interface where users can enter a movie title and view recommendations. It also incorporates a custom background image and modern styling.

6. Conclusion
This project successfully demonstrates the creation of a content-based movie recommendation system. By leveraging TF-IDF vectorization and the KNN algorithm, the system effectively identifies movies that are similar in content. The integration of critic reviews and metadata ensures that the recommendations are both diverse and relevant.

Future Enhancements:
Hybrid Recommendation Approach: Incorporate collaborative filtering to blend content-based and user-based recommendations.
User Feedback Integration: Allow users to rate recommendations to refine and personalize future suggestions.
Scalability Improvements: Optimize the model further to handle larger datasets and real-time updates.
