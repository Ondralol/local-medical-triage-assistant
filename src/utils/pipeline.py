### 



### RAG SETUP

import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer


# Load the FAISS index, embeddings, and metadata
index = faiss.read_index("RAG/my_index.faiss")
embeddings = np.load("RAG/embeddings.npy")
with open("RAG/metadata.pkl", "rb") as f:
    filtered = pickle.load(f)


# Model used to create the embedding of the query
model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")

# Function to query prompt
def query_rag(symptoms, k = 5):
    query_embedding = model.encode([symptoms]).astype("float32")
    D, I = index.search(query_embedding, k=k)
    results = filtered.iloc[I[0]]
    return results

"""

### BM25 SETUP

import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")

stop_words = set(stopwords.words("english"))

def clean_text_BM25(text):
    # If it's not string type
    if not isinstance(text, str):
        return ""

    text = text.lower() 
    text = re.sub(r"\s+", " ", text)  # Replace weird spacing with normal space
    
    # Remove cross reference sentences
    text = re.sub(r"(Read more|More information)[^.]*\.", "", text, flags=re.IGNORECASE)

    # Remove URLs
    text = re.sub(r"www\.\S+", "", text)

    text = text.strip() # Remove leading and trailing white spaces

    # tokenize
    tokens = word_tokenize(text)

    # remove stopwords
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)

def query_bm25(symptoms, k=5):
    # apply the same cleaning as corpus
    cleaned = clean_text_BM25(symptoms)
    tokens = cleaned.split()
    scores = bm25.get_scores(tokens)
    top_k = scores.argsort()[-k:][::-1]
    return filtered.iloc[top_k]



### COMBINED SETUP 

import pandas as pd

def combined_query(symptoms, k=5):
    res_rag = query_rag(symptoms, k)
    res_bm25 = query_bm25(symptoms, k)

    # Combine and drop duplicate rows
    combined = pd.concat([res_rag, res_bm25]).drop_duplicates(subset="question")
    return combined
"""

query_rag("I have a sore throat and a fever")


