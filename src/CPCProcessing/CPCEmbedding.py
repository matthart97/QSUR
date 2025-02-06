import torch
from transformers import BertModel, BertTokenizer
import pandas as pd
import pickle
from scipy.spatial.distance import cosine
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to embed descriptions using BERT
def get_bert_embeddings(texts):
    model.eval()  # Set model to evaluation mode
    embeddings = []
    with torch.no_grad():  # No need to compute gradients
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
            # Get the embeddings from the last hidden state (mean pooling)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return embeddings

# Function to load embeddings from a file
def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

# Function to compute cosine similarity between two embeddings
def cosine_similarity(embedding1, embedding2):
    # Cosine similarity is the inverse of cosine distance
    return 1 - cosine(embedding1, embedding2)


df = pd.read_csv('CPCDescriptions.csv')
CPCs = df['CPCDescription'].unique()


def get_bert_embeddings_with_labels(texts, labels):
    model.eval()
    embeddings_dict = {}
    
    with torch.no_grad():
        for text, label in zip(texts, labels):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings_dict[label] = embedding
    
    return embeddings_dict

# Create DataFrame from embeddings
def create_embeddings_df(embeddings_dict):
    # Convert dictionary to DataFrame
    df = pd.DataFrame({
        'label': list(embeddings_dict.keys()),
        'embedding': list(embeddings_dict.values())
    })
    return df


labels = CPCs
embeddings_dict = get_bert_embeddings_with_labels(CPCs, labels)

# Create DataFrame
df_embeddings = create_embeddings_df(embeddings_dict)

df_embeddings['code'] = df['CPCcode']

df_embeddings.to_pickle('CPCEmbeddings.pkl')