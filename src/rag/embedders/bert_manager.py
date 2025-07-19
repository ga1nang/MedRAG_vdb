import torch
from transformers import AutoTokenizer, AutoModel

class BERTManger:
    def __init__(self, model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

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