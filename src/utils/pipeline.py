import pandas as pd
import os
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./models/sentence_transformers"

### RAG SETUP

import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# bert
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


id2label = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

class Pipeline:
    def __init__(self):
        self.setup_dataset()
        self.setup_rag()
        self.setup_bm25()
        self.setup_bert()

    def setup_dataset(self):
        self.df = pd.read_csv("dataset/medquad_filtered.csv")


    def setup_rag(self):
        # Load the FAISS index, embeddings, and metadata
        self.index = faiss.read_index("RAG/my_index.faiss")
        self.embeddings = np.load("RAG/embeddings.npy")

        # Model used to create the embedding of the query
        self.model_rag = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
        

    # Function to query prompt
    def query_rag(self, symptoms, k = 5):
        query_embedding = self.model_rag.encode([symptoms]).astype("float32")
        D, I = self.index.search(query_embedding, k=k)
        results = self.df.iloc[I[0]]
        return results


    def setup_bm25(self):
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("punkt_tab")

        self.stop_words = set(stopwords.words("english"))

        with open("BM25/bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)


    def clean_text_BM25(self, text):
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
            if word not in self.stop_words
        ]

        return " ".join(tokens)

    def query_bm25(self, symptoms, k=5):
        # apply the same cleaning as corpus
        cleaned = self.clean_text_BM25(symptoms)
        tokens = cleaned.split()
        scores = self.bm25.get_scores(tokens)
        top_k = scores.argsort()[-k:][::-1]
        return self.df.iloc[top_k]



    def first_sentence(self, text, max_chars=150):
        if not isinstance(text, str):
            return ""
        return text.split(".")[0].strip()[:max_chars]

    def combined_query_processed(self, symptoms, k=3, max_length=200, disease_name = True):
        res_rag = self.query_rag(symptoms, 2)
        res_bm25 = self.query_bm25(symptoms, 5)

        # Combine and drop duplicate rows
        combined = pd.concat([res_rag, res_bm25]).drop_duplicates(subset="question")
        focus_areas = []
        symptoms = []
        for _, row in combined.iterrows():
            focus_areas.append(row["focus_area"])
            symptoms.append(self.first_sentence(row["answer"], max_length)) # TODO: answer can be replaced with text_for_rag or text_for_bm25

        string_result = []
        for i in range(len(combined)):
            if disease_name:
                string_result.append(f"Disease: {focus_areas[i]}, symptoms: {symptoms[i]}.")
            else:
                string_result.append(f"symptoms: {symptoms[i]}.")
            
        return "".join(string_result)
    
    def setup_bert(self):
        MODEL_PATH = "models/bert"

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        self.model.eval()
        torch.no_grad()


    def bert_inference(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True)
        

        with torch.no_grad():
            outputs = self.model(**inputs)

            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1)
            urgency = id2label[pred.item()]
            #print(f"Urgency: {urgency}")

            return urgency



if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.bert_inference("vomitting, bleeding, feeling sick")
    pipeline.bert_inference("cannot breath, coughing up blood")
    pipeline.bert_inference("tired, cough, runny nose")


    
    #print(pipeline.combined_query_processed("I have a sore throat and a fever"))


