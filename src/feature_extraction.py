import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm
import pandas as pd

class BertFeatureExtractor:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        print(f"BERT model loaded on {self.device}")
        
    def get_bert_embeddings(self, texts, batch_size=32):
        """
        Extract BERT embeddings for a list of texts.
        """
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting BERT embeddings"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize the texts
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # Move to device
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            # Disable gradient calculation for inference
            with torch.no_grad():
                outputs = self.model(**encoded_input)
            
            # Get the [CLS] token embeddings
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.extend(embeddings)
        
        return np.array(all_embeddings)

    def extract_features(self, movies_df):
        """
        Extract BERT features from movie data.
        """
        if 'text_for_bert' not in movies_df.columns:
            raise ValueError("DataFrame must contain 'text_for_bert' column")
        
        texts = movies_df['text_for_bert'].tolist()
        embeddings = self.get_bert_embeddings(texts)
        
        # Create a DataFrame with embeddings
        embeddings_df = pd.DataFrame(
            embeddings, 
            index=movies_df.index
        )
        
        # Add movie_id and title columns for reference
        embeddings_df['movie_id'] = movies_df['movie_id']
        embeddings_df['title'] = movies_df['title']
        
        return embeddings_df