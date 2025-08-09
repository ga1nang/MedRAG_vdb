import torch
import os
import spaces

from functools import lru_cache
from colpali_engine.models import ColPali, ColQwen2, ColQwen2Processor, ColIdefics3, ColIdefics3Processor
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device
from torch.utils.data import DataLoader
from typing import List, cast, Tuple
from tqdm import tqdm
from PIL import Image
from transformers import BitsAndBytesConfig
from src.rag.config import load_config

cfg = load_config()

@lru_cache(maxsize=1)
def load_colpali(quantized: bool, device: str = "cuda", model_name: str = "vidore/colpali-v1.3"):
    # Setup optional quantization
    bnb_cfg = None
    model = None
    processor = None
    if quantized:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Load ColPali
    if model_name == "vidore/colpali-v1.3":
        model = ColPali.from_pretrained(
            model_name,
            torch_dtype=None if quantized else torch.bfloat16,
            device_map=device,
            quantization_config=bnb_cfg,
        ).eval()
        processor = cast(
            ColPaliProcessor,
            ColPaliProcessor.from_pretrained(model_name)
        )

    # Load ColQwen2
    elif model_name == "vidore/colqwen2-v1.0":
        model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=None if quantized else torch.bfloat16,
            device_map=device,
            quantization_config=bnb_cfg,
        ).eval()
        processor = cast(
            ColQwen2Processor,
            ColQwen2Processor.from_pretrained(model_name)
        )

    # Load ColSmol (Idefics3)
    elif model_name == "vidore/colSmol-500M":
        model = ColIdefics3.from_pretrained(
            model_name,
            torch_dtype=None if quantized else torch.bfloat16,
            device_map=device,
            quantization_config=bnb_cfg,
        ).eval()
        processor = cast(
            ColIdefics3Processor,
            ColIdefics3Processor.from_pretrained(model_name)
        )

    elif model_name == "vidore/colSmol-256M":
        model = ColIdefics3.from_pretrained(
            model_name,
            torch_dtype=None if quantized else torch.bfloat16,
            device_map=device,
            quantization_config=bnb_cfg,
        ).eval()
        processor = cast(
            ColIdefics3Processor,
            ColIdefics3Processor.from_pretrained(model_name)
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    return model, processor

class ColPaliManager:
    def __init__(self, quantized: bool, device: str = "cuda", model_name: str = "vidore/colpali-v1.3"):
        self.device = get_torch_device(device)

        self.model, self.processor = load_colpali(quantized=quantized, device=self.device, model_name= model_name)

    @spaces.GPU
    def get_images(self, paths: List[str]) -> List[Image.Image]:
        images = []
        for path in paths:
            try:
                if os.path.exists(path):
                    images.append(Image.open(path).convert("RGB"))
            except Exception as e:
                print(f"Warning: Failed to load image {path}: {e}")
        return images

    @spaces.GPU
    def process_images(self, image_paths: List[str], batch_size=cfg["vit_batch_size"]):
        print(f"Processing {len(image_paths)} images")
        images = self.get_images(image_paths)

        dataloader = DataLoader(
            dataset=ListDataset[str](images),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_images(x),
        )

        ds: List[torch.Tensor] = []
        for batch_doc in tqdm(dataloader):
            with torch.no_grad():
                batch_doc = {k: v.to(self.model.device) for k, v in batch_doc.items()}
                embeddings_doc = self.model(**batch_doc)
            ds.extend(embeddings_doc.to(self.device).unbind(0))

        ds_np = [d.float().cpu().numpy() for d in ds]
        return ds_np

    @spaces.GPU
    def process_text(self, text: str):
        print(f"Processing {len(text)} texts")

        with torch.no_grad():
            batch_query = self.processor.process_queries([text]).to(self.model.device)
            query_embedding = self.model(**batch_query)
        multivector_query = query_embedding[0].cpu().float().numpy().tolist()
        return multivector_query