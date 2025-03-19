import os
import argparse
import pandas as pd
import pickle
from src.data_processing import DataProcessor
from src.feature_extraction import BertFeatureExtractor
from src.model import MovieClusteringModel
from src.recommendation import MovieRecommender

def main(args):
    # Create directories if they don't exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, 'processed'), exist_ok=True)
    
    # Paths
    raw_data_path = os.path.join(args.data_dir, 'raw', 'movies.csv')
    processed_data_path = os.path.join(args.data_dir, 'processed', 'movies_processed.csv')
    embeddings_path = os.path.join(args.data_dir, 'processed', 'movie_embeddings.pkl')
    model_path = os.path.join(args.data_dir, 'processed', 'movie_clusters.pkl')
    
    # Initialize data processor
    data_processor = DataProcessor(raw_data_path)
    
    # Process data
    if not os.path.exists(processed_data_path) or args.force_preprocess:
        print("Processing data...")
        movies_df = data_processor.load_data()
        movies_df = data_processor.preprocess_data()
        data_processor.save_processed_data(processed_data_path)
    else:
        print("Loading preprocessed data...")
        movies_df = pd.read_csv(processed_data_path)
    
    # Extract features
    if not os.path.exists(embeddings_path) or args.force_extract:
        print("Extracting BERT features...")
        feature_extractor = BertFeatureExtractor()
        embeddings_df = feature_extractor.extract_features(movies_df)
        data_processor.save_object(embeddings_df, embeddings_path)
    else:
        print("Loading pre-extracted embeddings...")
        embeddings_df = data_processor.load_object(embeddings_path)
    
    # Train clustering model
    if not os.path.exists(model_path) or args.force_train:
        print("Training clustering model...")
        cluster_model = MovieClusteringModel(n_clusters=args.n_clusters)
        
        if args.optimize_clusters:
            optimal_k = cluster_model.find_optimal_clusters(embeddings_df, max_clusters=args.max_clusters)
            print(f"Using optimal number of clusters: {optimal_k}")
        
        cluster_model.train(embeddings_df)
        data_processor.save_object(cluster_model, model_path)
        
        # Analyze clusters
        cluster_analysis = cluster_model.analyze_clusters(movies_df, embeddings_df)
        
        # Add cluster labels to movies_df
        movies_df['cluster'] = cluster_analysis['movies_with_clusters']['cluster']
        data_processor.save_processed_data(processed_data_path)
        
        # Print cluster analysis
        print("\nCluster Analysis:")
        for cluster_id, top_genres in cluster_analysis['top_genres'].items():
            print(f"Cluster {cluster_id}: {', '.join([f'{genre} ({score:.2f})' for genre, score in top_genres])}")
    else:
        print("Loading pre-trained clustering model...")
        cluster_model = data_processor.load_object(model_path)
        
        # Ensure movies_df has cluster labels
        if 'cluster' not in movies_df.columns:
            # Predict clusters
            X = embeddings_df.drop(['movie_id', 'title'], axis=1, errors='ignore')
            movies_df['cluster'] = cluster_model.predict_clusters(X)
            data_processor.save_processed_data(processed_data_path)
    
    # Initialize recommender
    recommender = MovieRecommender(movies_df, embeddings_df, cluster_model)
    
    # Example recommendations
    if args.example:
        print("\nExample recommendations:")
        # Find a popular movie
        popular_movie = movies_df.iloc[0]
        movie_id = popular_movie['movie_id']
        print(f"Recommendations for movie: {popular_movie['title']}")
        
        print("\nBased on similarity:")
        similar_movies = recommender.recommend_by_similarity(movie_id, n_recommendations=5)
        print(similar_movies[['title', 'genres', 'similarity_score']])
        
        print("\nBased on clustering:")
        cluster_movies = recommender.recommend_by_cluster(movie_id, n_recommendations=5)
        print(cluster_movies[['title', 'genres']])
        
        print("\nHybrid recommendations:")
        hybrid_movies = recommender.recommend_hybrid(movie_id, n_recommendations=5)
        print(hybrid_movies[['title', 'genres', 'hybrid_score']])
    
    print("\nRecommendation system is ready.")
    
    # Interactive mode
    if args.interactive:
        while True:
            print("\nEnter a movie title (or 'q' to quit): ")
            title = input()
            
            if title.lower() == 'q':
                break
            
            movie = recommender.get_movie_by_title(title)
            if movie is None:
                print(f"No movie found with title containing '{title}'")
                continue
            
            print(f"\nFound movie: {movie['title']} ({movie['genres']})")
            print("Recommendations:")
            
            hybrid_movies = recommender.recommend_hybrid(movie['movie_id'], n_recommendations=10)
            for _, rec in hybrid_movies.iterrows():
                print(f"{rec['title']} ({rec['genres']}) - Score: {rec['hybrid_score']:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory for data files")
    parser.add_argument("--n_clusters", type=int, default=15, help="Number of clusters")
    parser.add_argument("--optimize_clusters", action="store_true", help="Find optimal number of clusters")
    parser.add_argument("--max_clusters", type=int, default=30, help="Maximum number of clusters to try")
    parser.add_argument("--force_preprocess", action="store_true", help="Force preprocessing of data")
    parser.add_argument("--force_extract", action="store_true", help="Force extraction of features")
    parser.add_argument("--force_train", action="store_true", help="Force training of model")
    parser.add_argument("--example", action="store_true", help="Show example recommendations")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    main(args)