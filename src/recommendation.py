import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

class MovieRecommender:
    def __init__(self, movies_df, embeddings_df, cluster_model=None):
        self.movies_df = movies_df
        self.embeddings_df = embeddings_df
        self.cluster_model = cluster_model
        
        # Extract embedding vectors (excluding movie_id and title)
        self.embeddings = self.embeddings_df.drop(['movie_id', 'title'], axis=1, errors='ignore').values
        
        # Create movie id to index mapping
        self.movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(self.movies_df['movie_id'])}
        
    def get_movie_index(self, movie_id):
        """Get the index of a movie by its ID."""
        return self.movie_id_to_index.get(movie_id)
    
    def get_movie_by_title(self, title):
        """Get a movie by its title."""
        matches = self.movies_df[self.movies_df['title'].str.contains(title, case=False)]
        if len(matches) == 0:
            return None
        return matches.iloc[0]
    
    def recommend_by_similarity(self, movie_id, n_recommendations=10):
        """
        Recommend movies based on cosine similarity of BERT embeddings.
        """
        movie_idx = self.get_movie_index(movie_id)
        if movie_idx is None:
            raise ValueError(f"Movie ID {movie_id} not found")
        
        # Get the movie's embedding vector
        movie_embedding = self.embeddings[movie_idx].reshape(1, -1)
        
        # Calculate cosine similarity with all movies
        similarities = cosine_similarity(movie_embedding, self.embeddings)
        
        # Get indices of similar movies (excluding the input movie)
        similar_indices = similarities[0].argsort()[::-1][1:n_recommendations+1]
        
        # Get movie details for the similar movies
        recommendations = self.movies_df.iloc[similar_indices]
        
        # Add similarity scores
        recommendations['similarity_score'] = similarities[0][similar_indices]
        
        return recommendations
    
    def recommend_by_cluster(self, movie_id, n_recommendations=10):
        """
        Recommend movies from the same cluster.
        """
        if self.cluster_model is None:
            raise ValueError("Cluster model not provided")
        
        movie_idx = self.get_movie_index(movie_id)
        if movie_idx is None:
            raise ValueError(f"Movie ID {movie_id} not found")
        
        # Get the movie's embedding vector
        movie_embedding = self.embeddings[movie_idx].reshape(1, -1)
        
        # Get the movie's cluster
        movie_cluster = self.cluster_model.predict(movie_embedding)[0]
        
        # Get all movies in the same cluster
        cluster_movies = self.movies_df[self.movies_df['cluster'] == movie_cluster]
        
        # Exclude the input movie
        cluster_movies = cluster_movies[cluster_movies['movie_id'] != movie_id]
        
        # If there are fewer movies in the cluster than requested, return all
        if len(cluster_movies) <= n_recommendations:
            return cluster_movies
        
        # Otherwise, randomly select n_recommendations movies
        return cluster_movies.sample(n_recommendations, random_state=42)
    
    def recommend_hybrid(self, movie_id, n_recommendations=10, similarity_weight=0.7):
        """
        Hybrid recommendation using both similarity and clustering.
        """
        if self.cluster_model is None:
            raise ValueError("Cluster model not provided")
        
        movie_idx = self.get_movie_index(movie_id)
        if movie_idx is None:
            raise ValueError(f"Movie ID {movie_id} not found")
        
        # Get similar movies based on embedding similarity
        similar_movies = self.recommend_by_similarity(movie_id, n_recommendations=n_recommendations*2)
        
        # Get the movie's cluster
        movie_embedding = self.embeddings[movie_idx].reshape(1, -1)
        movie_cluster = self.cluster_model.predict(movie_embedding)[0]
        
        # Boost score for movies in the same cluster
        similar_movies['cluster'] = self.movies_df.loc[similar_movies.index, 'cluster']
        similar_movies['cluster_match'] = (similar_movies['cluster'] == movie_cluster).astype(int)
        
        # Calculate hybrid score
        similar_movies['hybrid_score'] = (
            similarity_weight * similar_movies['similarity_score'] + 
            (1 - similarity_weight) * similar_movies['cluster_match']
        )
        
        # Sort by hybrid score and get top n recommendations
        recommendations = similar_movies.sort_values('hybrid_score', ascending=False).head(n_recommendations)
        
        return recommendations[['movie_id', 'title', 'genres', 'similarity_score', 'cluster_match', 'hybrid_score']]
    
    def recommend_by_genre(self, genres, n_recommendations=10):
        """
        Recommend movies based on genres.
        """
        if isinstance(genres, str):
            genres = [genres]
        
        # Filter movies that contain at least one of the specified genres
        matching_movies = self.movies_df[
            self.movies_df['genres_list'].apply(lambda x: any(genre in x for genre in genres))
        ]
        
        # If no matching movies found
        if len(matching_movies) == 0:
            return pd.DataFrame()
        
        # If there are fewer matching movies than requested
        if len(matching_movies) <= n_recommendations:
            return matching_movies
        
        # Otherwise, randomly select n_recommendations movies
        return matching_movies.sample(n_recommendations, random_state=42)