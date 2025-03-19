from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import pickle
from src.data_processing import DataProcessor
from src.feature_extraction import BertFeatureExtractor
from src.model import MovieClusteringModel
from src.recommendation import MovieRecommender

app = Flask(__name__)

# Define data paths
# index.html will be served from the "templates" directory (Flask's default folder for templates)
data_dir = "data"
processed_data_path = os.path.join(data_dir, "processed", "movies_processed.csv")
embeddings_path = os.path.join(data_dir, "processed", "movie_embeddings.pkl")
model_path = os.path.join(data_dir, "processed", "movie_clusters.pkl")

# Load processed movies dataframe
movies_df = pd.read_csv(processed_data_path)

# Load pre-extracted embeddings
with open(embeddings_path, "rb") as f:
    embeddings_df = pickle.load(f)

# Load trained clustering model
with open(model_path, "rb") as f:
    cluster_model = pickle.load(f)

# Initialize movie recommender
recommender = MovieRecommender(movies_df, embeddings_df, cluster_model)

@app.route("/")
def index():
    # Render the index.html file from the templates directory
    return render_template("index.html")

@app.route("/recommend", methods=["GET"])
def recommend():
    # Get movie title from GET parameters
    movie_title = request.args.get("movie")
    if not movie_title:
        return jsonify({"error": "No movie title provided"}), 400
    
    # Find matching movie; get_movie_by_title returns one match or None
    movie = recommender.get_movie_by_title(movie_title)
    if movie is None:
        return jsonify({"error": f"Movie with title containing '{movie_title}' not found"}), 404
    
    # Get similar movies recommendations
    similar_movies = recommender.recommend_by_similarity(movie["movie_id"], n_recommendations=5)
    recs = similar_movies[['title', 'genres', 'similarity_score']].to_dict(orient="records")
    
    return jsonify({"movie": movie["title"], "recommendations": recs})

if __name__ == "__main__":
    app.run(debug=True)