import torch
import os
import numpy as np

from functools import lru_cache
from tqdm import tqdm
from typing import List
from src.rag.utils.utils import preprocess_text_rerank, extract_text_from_pdf, truncate_text
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- CPU threading knobs ---
os.environ.setdefault("OMP_NUM_THREADS", "20")
os.environ.setdefault("MKL_NUM_THREADS", "20")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

@lru_cache(maxsize=1)
def load_medbert(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval() 
    return tok, model

class ReRankerManager:
    def __init__(self, model_name: str = "ncbi/MedCPT-Cross-Encoder"):
        self.tokenizer, self.model = load_medbert(model_name=model_name)
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def re_rank(self, query: str, vectordb_docs: List[str]) -> List[str]:
        query = truncate_text(preprocess_text_rerank(query), max_chars=256)
        docs = [vectordb_doc['original_file'] for vectordb_doc in vectordb_docs]
        articles = [extract_text_from_pdf(pdf_path=doc, max_pages=3, max_chars=256) for doc in docs]
        pairs = [[query, article] for article in articles]

        with torch.no_grad():
            encoded = self.tokenizer(
                pairs,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            )

            logits = self.model(**encoded).logits.squeeze(dim=1)  # shape: [len(docs)]
            scores = logits.tolist()

        # Pair each doc with its score
        scored_docs = list(zip(docs, scores))

        # Sort by score (highest first)
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return only the documents in new order
        return [doc for doc, _ in scored_docs[:5]]
