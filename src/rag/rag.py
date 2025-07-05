import os

from typing import List
from src.rag.utils.utils import encode_image
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
os.environ["HF_HOME"] = "/media/pc1/Ubuntu/Extend_Data/hf_models"

class Rag:
    def __init__(self):
        self.pipe = pipeline("image-text-to-text", model="google/medgemma-4b-it", token=hf_token)
        self.message = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a clinical expert in Tropical and Infectious diseases diagnosis.\
                    User will give you all observable signs, symptoms of a patient's condition, such as physical indicators, patient-reported \
                    symptoms, and measurable clinical data. You will be given the image which contain the useful or related information of user's query.\
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

        # Call pipeline (chat mode)
        outputs = self.pipe(
            text=messages,
            images=images,
            max_new_tokens=1024,          # keep it reasonable
            return_full_text=False
        )

        # Assistant reply is in ["generated_text"]
        return outputs[0]["generated_text"]