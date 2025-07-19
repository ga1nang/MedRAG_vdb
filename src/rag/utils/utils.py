import base64
import re
import string
import pandas as pd
from nltk import word_tokenize
from PIL import Image

def encode_image(image_path):
    image = Image.open(image_path)
    return image

def preprocess_text(text):
    """Clean and normalize text: remove parentheses, punctuation, lowercase, tokenize."""
    if pd.isna(text):
        return ''
    text = re.sub(r'\(.*?\)', '', text).strip()
    text = text.replace('_', ' ')
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return ' '.join(tokens)