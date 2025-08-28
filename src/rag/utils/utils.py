import base64
import re
import string
import pandas as pd
import json
import warnings
from nltk import word_tokenize
from PIL import Image
from typing import List, Dict, Optional

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


    
def _parse_json(payload: str) -> Dict[str, List[str]]:
    cleaned = re.sub(r"```(?:json)?", "", payload, flags=re.I)
    cleaned = cleaned.replace("```", "").strip()
    return json.loads(cleaned)


def _normalise(seq: List[str]) -> List[str]:
    """
    • Lower‑cases
    • Removes punctuation / symbols
    • Collapses multiple spaces to one
    • Strips leading / trailing spaces
    • Deduplicates and returns an alphabetically‑sorted list
    """
    cleaned = (
        re.sub(r"\s+", " ",                       # collapse whitespace
            re.sub(r"[^\w\s]", " ", item))     # strip punctuation
        .strip()
        .lower()
        for item in seq if item.strip()
    )
    return sorted(set(cleaned))

def _parse_final_json_from_scratchpad(payload: str) -> Dict[str, List[str]]:
    """
    Extract JSON from:
    1) <final_json>...</final_json>
    2) ```json ... ```
    3) first {...} object found anywhere
    """
    # 1) Strict <final_json> block
    m = re.search(r"<final_json>\s*(.*?)\s*</final_json>", payload, re.DOTALL | re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        try:
            return _parse_json(candidate)  # already strips ```...``` if present
        except json.JSONDecodeError:
            warnings.warn("Failed to decode JSON inside <final_json> block.")

    # 2) Fenced code block: ```json { ... } ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", payload, re.DOTALL | re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            warnings.warn("Failed to decode JSON inside fenced code block.")

    # 3) First JSON object anywhere
    m = re.search(r"\{[\s\S]*?\}", payload)
    if m:
        candidate = m.group(0).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # lenient cleanup: single→double quotes, remove trailing commas
            cleaned = re.sub(r"'", '"', candidate)
            cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
            try:
                return json.loads(cleaned)
            except Exception:
                warnings.warn("Failed to decode loose JSON candidate.")

    warnings.warn("No JSON found in model output.")
    return {"history": [], "symptoms": []}