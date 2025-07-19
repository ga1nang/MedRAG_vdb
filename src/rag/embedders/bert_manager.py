import torch
from transformers import AutoTokenizer, AutoModel

class BERTManger:
    def __init__(self, model_name="")