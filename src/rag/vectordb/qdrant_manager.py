"""
A minimal wrapper around the qdrant-client SDK providing:
- Automatic collection creation (with configurable vector size & metric)
- Upsert of vectors + payloads
- Nearest-neighbour search
"""
import uuid
import numpy as np

from src.rag.config import load_config
from pathlib import Path
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models

# Load config
cfg = load_config()

class QdrantManager:
    """
    Conveniece wrapper for an embedded Qdrant instance
    (data lives in a local folder)
    """
    def __init__(
        self,
        collection_name: str = cfg["vector_db"]["collection_name"],
        create_collection: bool = True,
        vector_size: int = 128,
        metric: Distance = Distance.COSINE,
    ): 
        # Make sure parent dirs exist
        # self.db_path = Path(db_path).expanduser().resolve()
        # self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Init Qdrant "server" in-process
        self.client = QdrantClient(
           host="localhost",
            grpc_port=6334,
            prefer_grpc=True, 
            timeout=600,
        )        
        self.collection_name = collection_name
        
        # One-time collection creation
        if create_collection and not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                on_disk_payload=True,  # store the payload on disk
                vectors_config=models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    on_disk=True, # move original vectors to disk
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(
                        always_ram=False  # keep only quantized vectors in RAM
                        ),
                    ),
                ),
            )
            print(f"[Qdrant] Created collection “{collection_name}”(dim={vector_size}, metric={metric.name})")
            
    def insert_images_data(self, data: List[Dict[str, Any]]) -> None:
        """
        Upsert a batch of (vector, filepath) pairs.

        Args:
            data: (List[Dict[str, Any]])
        """
        # print(type(data[0]))
        # print(data[0])
        points = [
            PointStruct(
                id = uuid.uuid4().int >> 64,
                vector=record["colbert_vecs"],
                payload={"filepath": record["filepath"], "original_file": record['original_file']},
            )
            for record in data
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        print(f"Qdrant Upserted {len(points)} points")
        
    def search(
        self, 
        query_vec: List[float],
        top_k: int = cfg["retrieval_top_k"]
    ) -> List[Any]:
        """
        Return nearest neighbours as a list of (filepath, distance) tuples.

        Args:
            query_vec (np.ndarray | List[float]): Embedding of query
            top_k (int, optional): _description_. Defaults to cfg["retrieval_top_k"].
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vec,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "filepath": result.payload["filepath"],
                "original_file": result.payload["original_file"],
                "score": result.score,
            }
            for result in results
        ]
        
