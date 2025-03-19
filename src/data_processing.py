import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import re
import nltk
from nltk.corpus import stopwords
import pickle

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.stop_words = set(stopwords.words('english'))
        self.movies_df = None
        
    def load_data(self):
        """Load the movie dataset."""
        try:
            # Try auto-detecting the separator using the Python engine
            self.movies_df = pd.read_csv(self.data_path, sep=None, engine='python')
        except UnicodeDecodeError:
            self.movies_df = pd.read_csv(self.data_path, encoding='latin-1', sep=None, engine='python')
        
        print(f"Loaded {len(self.movies_df)} movies from {self.data_path}")
        return self.movies_df
    
    def preprocess_data(self):
        """Preprocess the movie data."""
        if self.movies_df is None:
            self.load_data()
        
        # Extract year from title
        self.movies_df['year'] = self.movies_df['title'].str.extract(r'\((\d{4})\)').astype('float')
        
        # Clean title (remove year)
        self.movies_df['clean_title'] = self.movies_df['title'].str.replace(r'\s*\(\d{4}\)\s*', '', regex=True)
        
        # Convert genres to list
        self.movies_df['genres_list'] = self.movies_df['genres'].str.split('|')
        
        # Create a text field combining title and genres for BERT
        self.movies_df['text_for_bert'] = self.movies_df.apply(
            lambda x: f"{x['clean_title']} {' '.join(x['genres_list'])}", 
            axis=1
        )
        
        # Clean text
        self.movies_df['text_for_bert'] = self.movies_df['text_for_bert'].apply(self._clean_text)
        
        print("Data preprocessing completed")
        return self.movies_df
    
    def _clean_text(self, text):
        """Clean text by removing special characters, numbers, and stopwords."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove stopwords
        tokens = nltk.word_tokenize(text)
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        
        return ' '.join(filtered_tokens)
    
    def create_genre_matrix(self):
        """Create a one-hot encoded matrix for genres."""
        if 'genres_list' not in self.movies_df.columns:
            self.preprocess_data()
        
        mlb = MultiLabelBinarizer()
        genres_matrix = mlb.fit_transform(self.movies_df['genres_list'])
        genres_df = pd.DataFrame(genres_matrix, columns=mlb.classes_)
        
        # Add movie_id and title for reference
        genres_df['movie_id'] = self.movies_df['movie_id']
        genres_df['title'] = self.movies_df['title']
        
        return genres_df
    
    def save_processed_data(self, output_path):
        """Save preprocessed data to disk."""
        if self.movies_df is None:
            self.preprocess_data()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.movies_df.to_csv(output_path, index=False)
        print(f"Saved preprocessed data to {output_path}")
        
    def save_object(self, obj, filename):
        """Save an object using pickle."""
        with open(filename, 'wb') as file:
            pickle.dump(obj, file)
        print(f"Saved object to {filename}")
        
    def load_object(self, filename):
        """Load an object using pickle."""
        with open(filename, 'rb') as file:
            obj = pickle.load(file)
        print(f"Loaded object from {filename}")
        return obj