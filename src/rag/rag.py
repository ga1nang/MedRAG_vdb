import os
# These lines must come BEFORE importing torch or transformers
# os.environ["PYTORCH_ENABLE_SDPA"] = "0"
# os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"
import torch

from google import genai
from google.genai import types
from functools import lru_cache
from typing import List, Dict, Optional
from src.rag.utils.utils import encode_image, _parse_final_json_from_scratchpad, _normalise
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
os.environ["HF_HOME"] = "/media/pc1/Ubuntu/Extend_Data/hf_models"

def load_gemini_client():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return client

def _pil_to_gemini_part(img: "PIL.Image.Image", mime: str = "image/png"):
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return types.Part.from_bytes(buf.getvalue(), mime_type=mime)

def _system_text_from_self_message(self) -> str:
    if self.message and self.message[0].get("role") == "system":
        parts = self.message[0].get("content", [])
        return "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")
    return ""

def _bnb_cfg(quantize: bool, qtype: str):
    if not quantize:
        return None
    if qtype == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif qtype == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"quantization_type must be '4bit' or '8bit', got {qtype!r}")

def _is_vlm_processor(proc) -> bool:
    return hasattr(proc, "image_processor") or hasattr(proc, "feature_extractor")

@lru_cache(maxsize=1)
def load_medgemma(model_name: str, quantize: bool, quantization_type: str):
    bnb = _bnb_cfg(quantize, quantization_type)

    # always try AutoProcessor first (works for VLMs, and often for text LLMs too)
    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
    
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb,
        trust_remote_code=True,
    )

    if _is_vlm_processor(processor):
        # ✅ VLM path (e.g., google/gemma-3-27b-it, google/medgemma-4b-it)
        return pipeline(
            task="image-text-to-text",
            model=model,
            processor=processor,          # <-- single processor is required
            return_full_text=False,
        )
    else:
        # Fallback: text-only models (kept for completeness)
        tok = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        return pipeline(
            task="text-generation",
            model=model,
            tokenizer=tok,
            return_full_text=False,
        )

class Rag:
    def __init__(self, model_name: str = "google/medgemma-4b-it", quantize: bool = False, quantization_type: str = "4bit"):
        """
        Args:
            quantize (bool): Whether to enable quantization.
            quantization_type (str): "4bit" or "8bit".
        """
        self.model_name = model_name
        if self.model_name[:6] == "gemini":
            self.client = load_gemini_client()
        else:
            # Build the pipeline with selected options
            self.pipe = load_medgemma(model_name = model_name, quantize=quantize, quantization_type=quantization_type)

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
        images = [encode_image(p) for p in images_path]

        if self.model_name[:6] == "gemini":
            # --- GEMINI: include system + user (images first, then text) ---
            from google.genai import types as gtypes
            system_text = self._system_text_from_self_message()

            system_content = gtypes.Content(
                role="system",
                parts=[gtypes.Part.from_text(system_text)]
            )

            user_parts = [ _pil_to_gemini_part(img) for img in images ]
            user_parts.append(gtypes.Part.from_text(query))

            user_content = gtypes.Content(role="user", parts=user_parts)

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[system_content, user_content],
            )
            return response.text

        # --- HF pipeline path: keep your existing chat format ---
        user_msg = {
            "role": "user",
            "content": ([{"type": "image"} for _ in images] + [{"type": "text", "text": query}]),
        }
        messages = self.message + [user_msg]

        with torch.inference_mode():
            outputs = self.pipe(
                text=messages,
                images=images,
                max_new_tokens=2048,
                return_full_text=False,
                do_sample=False,
                num_beams=1,
                use_cache=False,
            )
        result = outputs[0]["generated_text"]
        del outputs
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return result
    
    def feature_decomposition(
        self,
        query: str,
        images_path: Optional[List[str]] = None,
        max_new_tokens: int = 2048,
    ) -> Dict[str, List[str]]:
        images_path = images_path or []
        images = [encode_image(p) for p in images_path]
        messages = self._build_chat(query, images)  # HF chat format (for HF path)

        if self.model_name[:6] == "gemini":
            # --- Build Gemini contents using the *feature decomposition* system prompt ---
            from google.genai import types as gtypes
            sys_text = self._load_feature_decomposition_prompt()
            system_content = gtypes.Content(role="system", parts=[gtypes.Part.from_text(sys_text)])

            user_parts = [ _pil_to_gemini_part(img) for img in images ]
            user_parts.append(gtypes.Part.from_text(query))
            user_content = gtypes.Content(role="user", parts=user_parts)

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[system_content, user_content],
            )
            raw_output = response.text
        else:
            with torch.inference_mode():
                outputs = self.pipe(
                    text=messages,
                    images=images,
                    max_new_tokens=max_new_tokens,
                    return_full_text=False,
                    do_sample=False,
                    num_beams=1,
                    use_cache=False,
                )
            raw_output = outputs[0]["generated_text"]
            del outputs
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        result = _parse_final_json_from_scratchpad(raw_output)
        return {
            "history":  _normalise(result.get("history",  [])),
            "symptoms": _normalise(result.get("symptoms", [])),
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
