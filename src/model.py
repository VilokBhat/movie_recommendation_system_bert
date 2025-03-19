import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class MovieClusteringModel:
    def __init__(self, n_clusters=10, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        
    def find_optimal_clusters(self, embeddings, max_clusters=30):
        """
        Find the optimal number of clusters using the silhouette score.
        """
        # Get only the embedding columns (exclude movie_id and title)
        X = embeddings.drop(['movie_id', 'title'], axis=1, errors='ignore')
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        silhouette_scores = []
        k_values = range(2, max_clusters+1)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            print(f"For n_clusters = {k}, the silhouette score is {silhouette_avg:.4f}")
        
        # Find the optimal k
        optimal_k = k_values[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {optimal_k}")
        
        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, silhouette_scores, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs. Number of Clusters')
        plt.grid(True)
        plt.savefig('silhouette_scores.png')
        plt.close()
        
        self.n_clusters = optimal_k
        return optimal_k
    
    def train(self, embeddings):
        """
        Train the KMeans clustering model.
        """
        # Get only the embedding columns (exclude movie_id and title)
        X = embeddings.drop(['movie_id', 'title'], axis=1, errors='ignore')
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and fit KMeans
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.kmeans.fit(X_scaled)
        
        print(f"KMeans model trained with {self.n_clusters} clusters")
        return self.kmeans
    
    def predict_clusters(self, embeddings):
        """
        Predict clusters for the given embeddings.
        """
        if self.kmeans is None:
            raise ValueError("Model has not been trained yet")
        
        # Get only the embedding columns (exclude movie_id and title)
        X = embeddings.drop(['movie_id', 'title'], axis=1, errors='ignore')
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Predict clusters
        clusters = self.kmeans.predict(X_scaled)
        
        return clusters
    
    def analyze_clusters(self, movies_df, embeddings):
        """
        Analyze the clusters and generate insights.
        """
        if self.kmeans is None:
            raise ValueError("Model has not been trained yet")
        
        # Get only the embedding columns (exclude movie_id and title)
        X = embeddings.drop(['movie_id', 'title'], axis=1, errors='ignore')
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Predict clusters
        clusters = self.kmeans.predict(X_scaled)
        
        # Add cluster labels to movies dataframe
        movies_with_clusters = movies_df.copy()
        movies_with_clusters['cluster'] = clusters
        
        # Get cluster centers
        centers = self.kmeans.cluster_centers_
        
        # Analyze genre distribution within clusters
        cluster_genres = {}
        for cluster_id in range(self.n_clusters):
            cluster_movies = movies_with_clusters[movies_with_clusters['cluster'] == cluster_id]
            
            # Get all genres in this cluster
            all_genres = []
            for genres in cluster_movies['genres_list']:
                all_genres.extend(genres)
            
            # Count genre occurrences
            genre_counts = pd.Series(all_genres).value_counts()
            total_movies = len(cluster_movies)
            
            # Normalize by number of movies in cluster
            genre_distribution = genre_counts / total_movies
            
            cluster_genres[cluster_id] = genre_distribution.to_dict()
        
        # Get top genres for each cluster
        top_genres = {}
        for cluster_id, genres in cluster_genres.items():
            sorted_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)
            top_genres[cluster_id] = sorted_genres[:5]  # Top 5 genres
        
        # Visualize cluster sizes
        plt.figure(figsize=(12, 6))
        sns.countplot(x='cluster', data=movies_with_clusters)
        plt.title('Cluster Size Distribution')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Movies')
        plt.savefig('cluster_sizes.png')
        plt.close()
        
        return {
            'movies_with_clusters': movies_with_clusters,
            'cluster_genres': cluster_genres,
            'top_genres': top_genres,
            'centers': centers
        }