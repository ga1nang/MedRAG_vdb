import torch
import os
import numpy as np

from functools import lru_cache
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# --- CPU threading knobs ---
os.environ.setdefault("OMP_NUM_THREADS", "20")
os.environ.setdefault("MKL_NUM_THREADS", "20")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

@lru_cache(maxsize=1)
def load_medbert(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval() 
    return tok, model

class BERTManger:
    def __init__(self, model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"):
        self.tokenizer, self.model = load_medbert(model_name=model_name)
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def meanpooling(self, output, mask):
        embeddings = output[0] # First element of model_output contains all token embeddings
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    
    def generate_embedding(self, query):
        # Tokenize sentences
        inputs = self.tokenizer(query, padding=True, truncation=True, return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            output = self.model(**inputs)
        # Perform pooling. In this case, mean pooling.
        embeddings = self.meanpooling(output, inputs['attention_mask'])
        return embeddings
    
    def get_symptom_embeddings(self, symptom_nodes, save_path):
        """Generate or load embeddings for symptom nodes"""
        embeddings_path = os.path.join(save_path, 'KG_embeddings.npy')
        if os.path.exists(embeddings_path):
            print("load existing embeddings...")
            return np.load(embeddings_path)
        else:
            print("generate new embeddings...")
            symptom_embeddings = []
            # Generate embedding for each symptom
            for symptom in tqdm(symptom_nodes):
                symtom_embedding = self.generate_embedding(symptom)
                symptom_embeddings.append(symtom_embedding)
            # Save the embeddings for other use
            np.save(embeddings_path, symptom_embeddings)
            return np.array(symptom_embeddings)