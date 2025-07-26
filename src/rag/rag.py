import os
# These lines must come BEFORE importing torch or transformers
# os.environ["PYTORCH_ENABLE_SDPA"] = "0"
# os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"
import torch
import re
import json
import warnings

from typing import List, Dict, Optional
from src.rag.utils.utils import encode_image
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
os.environ["HF_HOME"] = "/media/pc1/Ubuntu/Extend_Data/hf_models"

class Rag:
    def __init__(self):
        self.pipe = pipeline(
            "image-text-to-text", 
            model="google/medgemma-4b-it", 
            torch_dtype=torch.bfloat16,
            token=hf_token,
            device_map="auto"
        )
        self.message = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a clinical expert in Tropical and Infectious diseases diagnosis.\
                    User will give you all observable signs, symptoms of a patient's condition, such as physical indicators, patient-reported \
                    symptoms, and measurable clinical data. You will be given  useful or related information of user's query retrieving from vector database and knowledge graph.\
                    Base on that, giving the diagnosis for the patient."}]
            }
        ]
    
    def get_answer_from_medgemma(self, query: str, images_path: List[str]) -> str:
        # Load images
        images = [encode_image(p) for p in images_path]    # PIL.Image.Image objects

        # Build the messages list with *image placeholders*
        messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image"} for _ in images] +   # one placeholder per image
                    [{"type": "text", "text": query}]
                ),
            }
        ]
        messages = self.message + messages

        # Call pipeline (chat mode)
        outputs = self.pipe(
            text=messages,
            images=images,
            max_new_tokens=1024,          # keep it reasonable
            return_full_text=False
        )

        # Assistant reply is in ["generated_text"]
        return outputs[0]["generated_text"]
    
    def feature_decomposition(
        self,
        query: str,
        images_path: Optional[List[str]] = None,
        max_new_tokens: int = 256,
    ) -> Dict[str, List[str]]:
        """
        Returns a dict with two keys:
            - 'history':  list of past or background conditions
            - 'symptoms': list of current, observable findings

        The LLM is asked to output *strict JSON*. We post‑process and
        normalise everything to slowercase, plain words; no punctuation for direct KG ingestion.
        """
        images_path = images_path or []

        # ---------- 1. prepare inputs ---------- #
        images_path = images_path or []
        images = [encode_image(p) for p in images_path] if images_path else []

        messages = self._build_chat(query, images)
        # ---------- 2. call the model ---------- #
        raw = self.pipe(
            text=messages,
            images=images,
            max_new_tokens=max_new_tokens,
            return_full_text=False
        )[0]["generated_text"]
        # ---------- 3. post‑process ---------- #
        try:
            result = self._parse_json(raw)
        except (json.JSONDecodeError, ValueError):
            warnings.warn("JSON parse failed – retrying with stricter prompt")
            messages[0]["content"][0]["text"] += (
                "\nIf nothing is present, return: {\"history\": [], \"symptoms\": []}"
            )
            raw = self.pipe(
                messages=messages,
                images=images if images else None,
                max_new_tokens=max_new_tokens,
                return_full_text=False,
            )[0]["generated_text"]
            result = self._parse_json(raw)
        # ---------- 4. normalise + deduplicate ---------- #
        return {
            "history":  self._normalise(result.get("history",  [])),
            "symptoms": self._normalise(result.get("symptoms", [])),
        }
    
    def _build_chat(self, query: str, images):
        """
        Returns a list‑of‑dicts in the HuggingFace chat format.
        If `images` is empty, no image placeholders are inserted.
        """
        sys_msg = {
            "role": "system",
            "content": [{"type": "text", "text": self._load_feature_decomposition_prompt()}],
        }
        user_content = [{"type": "text", "text": query}]
        if images:                                              # only add placeholders if any
            user_content = [{"type": "image"} for _ in images] + user_content
        return [sys_msg, {"role": "user", "content": user_content}]
    
    def _load_feature_decomposition_prompt(self) -> str:
        return (
            "You are an expert clinical scribe and medical language model. "
            "Your task is to extract structured medical features from the USER text. "
            "Follow the instructions below EXACTLY. Any deviation will result in strict penalties.\n\n"
            "1. Identify and separate ONLY two types of information:\n"
            "   - 'history': past medical conditions (chronic diseases, prior infections, surgeries, lifestyle factors, risk factors).\n"
            "   - 'symptoms': current issues (patient-reported complaints, objective clinical signs, abnormal findings).\n\n"
            "2. Your response MUST be STRICTLY valid JSON, following these rules:\n"
            "   - The JSON must have EXACTLY this structure: {\"history\": [...], \"symptoms\": [...]}.\n"
            "   - Use ONLY double quotes for keys and string values.\n"
            "   - Each array item must be a lowercase, plain word or phrase with no punctuation.\n"
            "   - NO extra text, NO explanations, NO reasoning, NO markdown, and NO code fences.\n"
            "   - If no history or symptoms are present, return EXACTLY: {\"history\": [], \"symptoms\": []}.\n"
            "   - The JSON must be complete, well-formed, and can be parsed directly by a JSON parser.\n\n"
            "3. STRICT PENALTY: If your response contains anything other than valid JSON in the exact format described, "
            "it will be considered a critical failure. You MUST comply with these rules in every case.\n\n"
            "Correct Examples:\n"
            "Example 1:\n"
            "{\"history\": [\"hypertension\", \"smoking\"], \"symptoms\": [\"shortness of breath\", \"dizziness\"]}\n\n"
            "Example 2:\n"
            "{\"history\": [], \"symptoms\": [\"fever\", \"rash\", \"joint pain\"]}\n\n"
            "Example 3:\n"
            "{\"history\": [\"diabetes\", \"kidney transplant\"], \"symptoms\": [\"abdominal pain\"]}\n\n"
            "Example 4 (empty):\n"
            "{\"history\": [], \"symptoms\": []}\n\n"
            "Wrong (will be penalized):\n"
            "```json\n{\"history\": [\"diabetes\"], \"symptoms\": [\"fever\"]}\n```\n"
            "or adding any explanations like 'the patient has diabetes'.\n\n"
            "Now, read the USER text carefully and respond ONLY with the JSON object as specified. "
            "Do NOT include any extra characters or commentary."
        )
    
    def _parse_json(self, payload: str) -> Dict[str, List[str]]:
            # strip code‑block fencing if the model wrapped output in ```json
            cleaned = re.sub(r"^```(?:json)?|```$", "", payload.strip(), flags=re.I | re.S)
            return json.loads(cleaned)

    def _normalise(self, seq: List[str]) -> List[str]:
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