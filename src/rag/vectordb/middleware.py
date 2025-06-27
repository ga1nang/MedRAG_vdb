"""
Glue code that ties together:
- PdfManager -> extracts images
- ColPaliManager -> embeds images & queries
- QdrantManager -> stores & searches vectors
"""

import hashlib
from pathlib import Path
from typing import List, Any

from rag.utils.pdf_manager import PdfManager
from rag.embedders.colpali_manager import ColPaliManager
from rag.vectordb.qdrant_manager import QdrantManager
from rag.config import load_config

# Load config
cfg = load_config()

class Middleware:
    """Main entry-point for index() and search() used by the app"""
    
    def __init__(self, user_id: str, create_collection: bool = True):
        # Init manager
        self.pdf_manager = PdfManager()
        self.colpali_manager = ColPaliManager()
        
        # Create Qdrant fodler
        qdrant_folder = Path(cfg["paths"]["db_path"])
        qdrant_folder.mkdir(exist_ok=True)
        
        # Start Qdrant wrapper
        self.db = QdrantManager(
            db_path=qdrant_folder,
            collection_name=cfg["vector_db"]["collection_name"],
            vector_size=cfg["vector_db"]["vector_size"],
            create_collection=create_collection,
        )
        
    def index(
        self,
        id: str,
        pdf_path: str,
    ) -> List[str]:
        print(f"[Middleware] Indexing {pdf_path}")
        
        # Extract page images
        image_paths = self.pdf_manager.save_images(
            id=id,
            pdf_path=pdf_path,
        )
        print(f"  • Saved {len(image_paths)} page images")
        
        # Bacth -> ColPali embeddings
        vectors = self.colpali_manager.process_images(image_paths)
        assert len(vectors) == len(image_paths)
        
        # Wrap for Qdrant
        db_payload = [
            {
                "colbert_vecs": vectors[i],
                "filepath": image_paths[i]
            }
            for i in range(len(image_paths))
        ]
        print(f"Inserting {len(db_payload)} vectors into Qdrant…")
        self.db.insert_images_data(db_payload)
        print("[Middleware] Indexing complete ✓\n")
        return image_paths
    
    def search(
        self, 
        queries: List[str], 
        top_k: int = cfg["retrieval_top_k"]
    ) -> List[Any]:
        """
        Embed each query with ColPali's text encoder, 
        then fetch top-k image pages from vector database.
        """    
        print(f"[Middleware] Received {len(queries)} query/queries")
        results = []
        
        for query in queries:
            print(f"  → {query!r}")
            query_vec = self.colpali_manager.process_text([query])[0]
            result = self.db.search(query_vec, top_k)
            print(f"    ↳ hits: {result}")
            results.append(result)
            
        return results