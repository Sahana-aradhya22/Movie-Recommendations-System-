from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import pickle
import os
import numpy as np
from difflib import get_close_matches  # For fuzzy title matching
import asyncio
from fastapi.staticfiles import StaticFiles


app = FastAPI()

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# Helper function to load pickled models safely
def load_pickle(file_path):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        raise RuntimeError(f"Error: Missing or corrupted file - {file_path}")
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Load preprocessed data
try:
    knn = load_pickle("models/knn_model.pkl")
    movie_indices = load_pickle("models/movie_indices.pkl")
    tfidf_matrix = load_pickle("models/tfidf_matrix.pkl")
    tfidf_vectorizer = load_pickle("models/tfidf_vectorizer.pkl")
    movies_df = pd.read_csv("data/processed_movies.csv")  # Use processed dataset

    # Ensure all titles are lowercase strings for case-insensitive search
    movie_indices_lower = {
        str(key).strip().lower(): value
        for key, value in movie_indices.items()
        if pd.notna(key)
    }
except Exception as e:
    raise RuntimeError(f"Failed to initialize the application: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/recommend", response_class=HTMLResponse)
async def recommend_movies(request: Request, title: str = Form(...)):
    try:
        title = title.strip().lower()  # Normalize input

        # Handle title mismatches using fuzzy matching
        if title not in movie_indices_lower:
            close_matches = get_close_matches(title, movie_indices_lower.keys(), n=1, cutoff=0.7)
            if close_matches:
                title = close_matches[0]  # Use the closest matching title
            else:
                return templates.TemplateResponse("index.html", {
                    "request": request, 
                    "error": f"Movie '{title}' not found. Try a different title."
                })

        # Get movie index
        idx = movie_indices_lower.get(title)
        if idx is None or idx >= tfidf_matrix.shape[0]:
            raise HTTPException(status_code=404, detail="Movie not found")

        # Find similar movies using KNN
        distances, indices = knn.kneighbors(tfidf_matrix[idx].reshape(1, -1), n_neighbors=10)

        # Ensure indices are within bounds
        valid_indices = [i for i in indices[0][1:] if i < len(movies_df)]
        if not valid_indices:
            return templates.TemplateResponse("index.html", {
                "request": request, 
                "error": "No valid recommendations found."
            })

        # Retrieve recommended movies
        recommended_movies = movies_df.iloc[valid_indices][['title', 'genre', 'director']]
        
        # Remove duplicate movie titles
        recommended_movies = recommended_movies.drop_duplicates(subset=['title'], keep='first')

        return templates.TemplateResponse("index.html", {
            "request": request,
            "movies": recommended_movies.to_dict(orient='records'),
            "title": title.capitalize()
        })
    
    except asyncio.CancelledError:
        print("Request was cancelled by the client")
        raise HTTPException(status_code=499, detail="Client cancelled the request")

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": f"An error occurred: {str(e)}"
        })
