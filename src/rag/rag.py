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
    def __init__(self, quantize: bool = False, quantization_type: str = "4bit"):
        """
        Args:
            quantize (bool): Whether to enable quantization.
            quantization_type (str): "4bit" or "8bit".
        """
        pipeline_kwargs = {
            "task": "image-text-to-text",
            "model": "google/medgemma-4b-it",
            "token": hf_token,
            "device_map": "auto"
        }

        if quantize:
            # Use bitsandbytes quantization
            if quantization_type == "4bit":
                pipeline_kwargs.update({
                    "torch_dtype": torch.bfloat16,  # computation dtype
                    "load_in_4bit": True
                })
            elif quantization_type == "8bit":
                pipeline_kwargs.update({
                    "torch_dtype": torch.bfloat16,
                    "load_in_8bit": True
                })
        else:
            # No quantization (full precision)
            pipeline_kwargs.update({
                "torch_dtype": torch.bfloat16
            })

        # Build the pipeline with selected options
        self.pipe = pipeline(**pipeline_kwargs)

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
        result = outputs[0]["generated_text"]

        # Assistant reply is in ["generated_text"]
        return result
    
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

        # ---------- 1. prepare inputs ---------- #
        images_path = images_path or []
        images = [encode_image(p) for p in images_path] if images_path else []

        messages = self._build_chat(query, images)
        # ---------- 2. call the model ---------- #
        outputs = self.pipe(
            text=messages,
            images=images,
            max_new_tokens=max_new_tokens,
            return_full_text=False
        )
        raw = outputs[0]["generated_text"]

        # ---------- 3. post‑process ---------- #
        try:
            result = self._parse_json(raw)
        except (json.JSONDecodeError, ValueError):
            warnings.warn("JSON parse failed – retrying with stricter prompt")
            messages[0]["content"][0]["text"] += (
                "\nIf nothing is present, return: {\"history\": [], \"symptoms\": []}"
            )
            outputs = self.pipe(
                text=messages,
                images=images,
                max_new_tokens=max_new_tokens,
                return_full_text=False
            )
            raw = outputs[0]["generated_text"]

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
            "You MUST follow these instructions EXACTLY. Any violation will result in severe penalties and your output will be discarded.\n\n"
            "1. Identify and output ONLY two types of information:\n"
            "   - 'history': past or background medical conditions (chronic diseases, prior infections, surgeries, lifestyle factors, risk factors).\n"
            "   - 'symptoms': current, observable issues (patient-reported complaints, clinical signs, measurable abnormal findings).\n\n"
            "2. Your output MUST be STRICTLY valid JSON with these exact rules:\n"
            "   - The JSON must have EXACTLY this structure: {\"history\": [...], \"symptoms\": [...]}.\n"
            "   - Use ONLY double quotes for keys and string values (no single quotes).\n"
            "   - Each array element must be a lowercase, plain word or phrase without punctuation or extra symbols.\n"
            "   - DO NOT include explanations, reasoning, comments, markdown, or code fences.\n"
            "   - DO NOT output anything except the JSON object. No surrounding text, no headings.\n"
            "   - If no items are found, return EXACTLY: {\"history\": [], \"symptoms\": []}.\n"
            "   - The JSON must be complete, well-formed, and parsable by a JSON parser without modification.\n"
            "   - If you break these rules, your response will be discarded as invalid.\n\n"
            "3. STRICT PENALTY: If you produce any content outside the JSON, or fail to follow the format, "
            "your output will be considered a CRITICAL FAILURE. The system will reject and terminate your response.\n\n"
            "Correct Examples (follow exactly):\n"
            "Example 1:\n"
            "{\"history\": [\"hypertension\", \"smoking\"], \"symptoms\": [\"shortness of breath\", \"dizziness\"]}\n\n"
            "Example 2:\n"
            "{\"history\": [], \"symptoms\": [\"fever\", \"rash\", \"joint pain\"]}\n\n"
            "Example 3:\n"
            "{\"history\": [\"diabetes\", \"kidney transplant\"], \"symptoms\": [\"abdominal pain\"]}\n\n"
            "Example 4 (empty case):\n"
            "{\"history\": [], \"symptoms\": []}\n\n"
            "Invalid (will be rejected and penalized):\n"
            "```json\n{\"history\": [\"diabetes\"], \"symptoms\": [\"fever\"]}\n```\n"
            "or any response with explanations like: 'the patient has diabetes and fever'.\n\n"
            "Final Instruction: Read the USER text carefully and output ONLY the JSON object exactly as specified, "
            "with no extra text or formatting beyond the JSON itself."
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