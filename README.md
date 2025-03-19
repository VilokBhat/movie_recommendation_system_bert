# Movie Recommendation System with BERT and Clustering

This project implements a movie recommendation system using BERT embeddings and clustering techniques. It takes a dataset of movies with their titles and genres, and provides recommendations based on semantic similarity and cluster membership.

## Project Structure

```
movie_recommendation_system/
│
├── data/
│   ├── raw/
│   │   └── movies.csv
│   └── processed/
│       ├── movie_embeddings.pkl
│       └── movie_clusters.pkl
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── feature_extraction.py
│   ├── model.py
│   └── recommendation.py
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── main.py
├── requirements.txt
└── README.md
```

## Features

- Data processing and feature extraction using BERT embeddings
- Clustering of movies based on their BERT embeddings
- Multiple recommendation strategies:
  - Similarity-based recommendations
  - Cluster-based recommendations
  - Hybrid recommendations combining both approaches
  - Genre-based recommendations
- Interactive mode for querying recommendations

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bert-movie-recommendation-system.git
cd movie-recommendation-system
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Place your movies.csv file in the data/raw/ directory.

## Usage

### Basic Usage

Run the main script:
```bash
python main.py
```

### Options

- `--data_dir`: Directory for data files (default: "data")
- `--n_clusters`: Number of clusters (default: 15)
- `--optimize_clusters`: Find optimal number of clusters using silhouette scores
- `--max_clusters`: Maximum number of clusters to try during optimization (default: 30)
- `--force_preprocess`: Force preprocessing of data
- `--force_extract`: Force extraction of features
- `--force_train`: Force training of model
- `--example`: Show example recommendations
- `--interactive`: Interactive mode for querying recommendations

### Example

```bash
python main.py --optimize_clusters --example --interactive
```

## How It Works

1. **Data Preprocessing**:
   - Extracts year from movie titles
   - Cleans and tokenizes text data
   - Converts genres to lists

2. **Feature Extraction**:
   - Uses BERT to generate embeddings for each movie
   - Combines title and genre information

3. **Clustering**:
   - Applies KMeans clustering to group similar movies
   - Optimizes the number of clusters using silhouette scores

4. **Recommendation**:
   - Similarity-based: Recommends movies with similar BERT embeddings
   - Cluster-based: Recommends movies from the same cluster
   - Hybrid: Combines similarity and cluster information

## Results

The system can identify semantic similarities between movies that go beyond simple genre matching. For example, it can recognize that "The Matrix" and "Inception" are similar sci-fi films with philosophical themes, even if they have different genre tags.

The cluster analysis reveals interesting patterns in the movie dataset, grouping films not just by genre but by themes, styles, and other latent characteristics captured by the BERT embeddings.

## Future Improvements

- Incorporate user ratings and preferences
- Implement collaborative filtering alongside content-based methods
- Develop a web interface for easier interaction
- Add more features like director, actors, and plot summaries