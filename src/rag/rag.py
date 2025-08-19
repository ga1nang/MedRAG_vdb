import os
# These lines must come BEFORE importing torch or transformers
# os.environ["PYTORCH_ENABLE_SDPA"] = "0"
# os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"
import torch
import re
import json
import warnings

from functools import lru_cache
from typing import List, Dict, Optional
from src.rag.utils.utils import encode_image
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
os.environ["HF_HOME"] = "/media/pc1/Ubuntu/Extend_Data/hf_models"

@lru_cache(maxsize=1)   
def load_medgemma(quantize: bool, quantization_type: str):
    kwargs = {
        "task": "image-text-to-text",
        "model": "google/medgemma-4b-it",
        "token": hf_token,
        "device_map": "cuda",
        "torch_dtype": torch.bfloat16,
    }
    if quantize:
        if quantization_type == "4bit":
            kwargs["load_in_4bit"] = True
        elif quantization_type == "8bit":
            kwargs["load_in_8bit"] = True
    return pipeline(**kwargs)   

class Rag:
    def __init__(self, quantize: bool = False, quantization_type: str = "4bit"):
        """
        Args:
            quantize (bool): Whether to enable quantization.
            quantization_type (str): "4bit" or "8bit".
        """
        # Build the pipeline with selected options
        self.pipe = load_medgemma(quantize=quantize, quantization_type=quantization_type)

        self.message = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self._load_system_prompt()}]
            }
        ]

    def _load_system_prompt(self) -> str:
        return """
            You are a clinician specialized in Tropical & Infectious Diseases.

            # Objective
            Evaluate a patient case (text + optional images) together with two context blocks:
            (1) retrieved document excerpts and (2) knowledge-graph facts. Use this information to arrive at the most likely diagnosis with clear clinical reasoning.

            # Reasoning Approach
            • Use System 1 (intuitive, pattern recognition) and System 2 (analytic, hypothesis testing).
            • Think step by step and structure your reasoning using these four stages:
            1) Information Gathering — key signs/symptoms and duration; epidemiologic context (travel, residence, occupation, exposures); relevant comorbidities/medications; red-flag features (e.g., hemorrhage, shock, AMS, jaundice).
            2) Hypothesis Generation — organize by anatomical system and by timing/progression; consider age and risk factors; generate a broad differential for tropical/infectious causes; apply illness scripts (e.g., dengue, malaria, leptospirosis, TB, rickettsioses, typhoid, chikungunya, Zika, acute HIV, etc.).
            3) Hypothesis Testing — identify defining vs. discriminatory features; propose point-of-care tests, labs, or imaging that would confirm or refute candidates; note potential diagnostic biases (anchoring, availability, framing) and how you mitigate them; mention early empiric considerations at a general level when appropriate.
            4) Reflection & Final Diagnosis — re-assess fit with expected disease course; consider multi-system disease or co-infection; state the most likely diagnosis and briefly outline initial supportive care priorities and warning signs to monitor.

            # Grounding
            • Prefer facts from the provided CONTEXT; if something is unknown, state it rather than invent it.

            # Output
            • Present your reasoning clearly with the four stage headings above.
            • Conclude with a succinct summary that includes these two labeled lines (for downstream parsing):
            WORKING_DIAGNOSIS: <your concise working diagnosis, include severity/stage if evident>
            DISEASE_NAME: <canonical disease name>
            • Use a professional, polite tone throughout.

            # Safety
            • Provide educational clinical reasoning; avoid personalized medical directives or drug dosing. Emphasize red-flags and advise urgent in-person evaluation when indicated.
        """
    
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
        print(messages)
        # Call pipeline (chat mode)
        outputs = self.pipe(
            text=messages,
            images=images,
            max_new_tokens=2048,
            return_full_text=False,
            do_sample=False,   # temperature is ignored
            num_beams=1,       # ensure pure greedy
        )
        result = outputs[0]["generated_text"]
        # Free VRAM
        del outputs
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # Assistant reply is in ["generated_text"]
        return result
    
    def feature_decomposition(
        self,
        query: str,
        images_path: Optional[List[str]] = None,
        max_new_tokens: int = 2048,
    ) -> Dict[str, List[str]]:
        """
        Extracts 'history' and 'symptoms' using the reliable scratchpad method.
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
            return_full_text=False,
            do_sample=False,   # temperature is ignored
            num_beams=1,       # ensure pure greedy
        )
        raw_output = outputs[0]["generated_text"]
        # print(f"Raw output: \n{raw_output}")
        # Free VRAM
        del outputs
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # ---------- 3. parse and normalise ---------- #
        # Use the new, dedicated parser for the scratchpad format
        result = self._parse_final_json_from_scratchpad(raw_output)
        
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
    
    # def _load_feature_decomposition_prompt(self) -> str:
    #     """
    #     Returns the optimized prompt for MedGemma (4B) to extract structured
    #     clinical features from free-text input. The model must output only
    #     a valid JSON object with "history" and "symptoms" keys.
    #     """
    #     return (
    #         "You are an automated medical data extraction engine. Your sole function is to parse clinical text "
    #         "and return a single, raw JSON object. Your output is fed directly into a program and will fail "
    #         "if not formatted precisely. Do not provide any explanation or conversational text.\n\n"

    #         "<INSTRUCTIONS>\n"
    #         "1.  **Analyze**: Carefully read the <USER_INPUT> to identify all relevant clinical information.\n"
    #         "2.  **Categorize**: Separate the information into exactly two categories:\n"
    #         "    - `history`: Chronic diseases, past surgeries, and relevant lifestyle/risk factors (e.g., smoking).\n"
    #         "    - `symptoms`: Current complaints, signs, or findings the patient is actively experiencing.\n"
    #         "3.  **Normalize**: Convert all extracted items into lowercase, short phrases. Remove all internal punctuation.\n"
    #         "4.  **Format**: Construct a single JSON object. The first character of your output must be `{` and the last must be `}`.\n"
    #         "</INSTRUCTIONS>\n\n"

    #         "<FORMATTING_RULES>\n"
    #         "-   **Output**: Must be ONLY a valid JSON object.\n"
    #         "-   **Keys**: Must be exactly `\"history\"` and `\"symptoms\"`.\n"
    #         "-   **Values**: Must be a JSON array of strings `[...]`.\n"
    #         "-   **Empty State**: If no items are found for a category, use an empty array `[]`. If the input is empty or irrelevant, output exactly `{\"history\": [], \"symptoms\": []}`.\n"
    #         "-   **Prohibited**: Do NOT include markdown (like ```json), comments, apologies, or any text outside the JSON structure.\n"
    #         "-   **Quotes**: Use only double quotes `\"` for all keys and string values.\n"
    #         "-   **Failure**: Any deviation from these rules constitutes a critical failure.\n"
    #         "</FORMATTING_RULES>\n\n"

    #         "<EXAMPLES>\n"
    #         "USER_INPUT: \"I have had type-2 diabetes since 2015 and a knee replacement, but now I'm coughing and feel feverish.\"\n"
    #         "OUTPUT: {\"history\": [\"type 2 diabetes\", \"knee replacement\"], \"symptoms\": [\"cough\", \"fever\"]}\n\n"

    #         "USER_INPUT: \"healthy teenager, sudden chest pain and shortness of breath after basketball.\"\n"
    #         "OUTPUT: {\"history\": [], \"symptoms\": [\"chest pain\", \"shortness of breath\"]}\n\n"

    #         "USER_INPUT: \"74-year-old smoker with hypertension, presents with weight loss and hemoptysis.\"\n"
    #         "OUTPUT: {\"history\": [\"smoking\", \"hypertension\"], \"symptoms\": [\"weight loss\", \"hemoptysis\"]}\n\n"

    #         "USER_INPUT: \"no medical issues, feeling great.\"\n"
    #         "OUTPUT: {\"history\": [], \"symptoms\": []}\n"
    #         "</EXAMPLES>\n\n"

    #         "Process the following input according to all rules.\n\n"
    #         "<USER_INPUT>"
    #     )

    def _load_feature_decomposition_prompt(self) -> str:
        """
        Returns the optimized prompt for MedGemma (4B) to extract structured
        clinical features from free-text input. This version uses a two-step
        process (scratchpad -> final output) to maximize reliability.
        """
        return (
            "You are a clinical data structuring tool. Your task is to follow a two-step process to extract information from the user's text.\n\n"
            
            "**STEP 1: Analyze in a Scratchpad**\n"
            "First, create a `<scratchpad>` block. Inside this block, you will:\n"
            "1.  Identify and list all `history` items (past/chronic conditions, surgeries, lifestyle factors).\n"
            "2.  Identify and list all `symptoms` items (current complaints).\n"
            "This is your workspace. Be descriptive here.\n\n"

            "**STEP 2: Generate Final JSON**\n"
            "After the scratchpad, create a `<final_json>` block. Inside this block, place ONLY the final, valid JSON object.\n"
            "-   The JSON must contain only the keys `\"history\"` and `\"symptoms\"`.\n"
            "-   All strings inside the JSON must be lowercase and contain no punctuation.\n"
            "-   If no items are found for a key, use an empty array `[]`.\n\n"

            "Your final response MUST contain both the `<scratchpad>` and the `<final_json>` blocks.\n\n"
            
            "--- EXAMPLES ---\n\n"
            
            "USER_INPUT: \"I have had type-2 diabetes since 2015 and a knee replacement, but now I'm coughing and feel feverish.\"\n"
            "OUTPUT:\n"
            "<scratchpad>\n"
            "History Items: type-2 diabetes, knee replacement.\n"
            "Symptom Items: coughing, feverish.\n"
            "Normalization: convert to lowercase and format.\n"
            "</scratchpad>\n"
            "<final_json>\n"
            "{\"history\": [\"type 2 diabetes\", \"knee replacement\"], \"symptoms\": [\"cough\", \"fever\"]}\n"
            "</final_json>\n\n"

            "USER_INPUT: \"74-year-old smoker with hypertension, presents with weight loss and hemoptysis.\"\n"
            "OUTPUT:\n"
            "<scratchpad>\n"
            "History Items: smoker, hypertension.\n"
            "Symptom Items: weight loss, hemoptysis.\n"
            "Normalization: convert to lowercase and format.\n"
            "</scratchpad>\n"
            "<final_json>\n"
            "{\"history\": [\"smoking\", \"hypertension\"], \"symptoms\": [\"weight loss\", \"hemoptysis\"]}\n"
            "</final_json>\n\n"

            "USER_INPUT: \"no medical issues, feeling great.\"\n"
            "OUTPUT:\n"
            "<scratchpad>\n"
            "History Items: None.\n"
            "Symptom Items: None.\n"
            "</scratchpad>\n"
            "<final_json>\n"
            "{\"history\": [], \"symptoms\": []}\n"
            "</final_json>\n\n"

            "--- TASK ---\n"
            "Process the following input according to the two-step process.\n\n"
            "USER_INPUT: "
        )

    
    def _parse_json(self, payload: str) -> Dict[str, List[str]]:
        cleaned = re.sub(r"```(?:json)?", "", payload, flags=re.I)
        cleaned = cleaned.replace("```", "").strip()
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
    
    def _parse_final_json_from_scratchpad(self, payload: str) -> Dict[str, List[str]]:
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
                return self._parse_json(candidate)  # already strips ```...``` if present
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
