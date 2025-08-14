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

def preprocess_text_rerank(text):
    """Clean and normalize text: remove parentheses, punctuation, lowercase, tokenize, remove newlines."""
    if pd.isna(text):
        return ''
    text = re.sub(r'\(.*?\)', '', text).strip()
    text = text.replace('_', ' ')
    text = text.replace('\n', ' ')  # remove newlines
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return ' '.join(tokens)


def extract_text_from_pdf(pdf_path, max_pages: int = 3, max_chars: int = 3000) -> str:
    """Extracts limited text from the first few pages of a PDF."""
    import fitz
    doc = fitz.open(pdf_path)
    all_text = ""
    for page_num in range(min(len(doc), max_pages)):
        text = doc[page_num].get_text("text").strip()
        all_text += f"\n--- Page {page_num + 1} ---\n{text}"
    doc.close()
    return truncate_text(all_text, max_chars)

def truncate_text(txt: str, max_chars: int) -> str:
    """Truncates text to a safe length for prompt."""
    return txt if len(txt) <= max_chars else txt[:max_chars] + "\n...[truncated]..."