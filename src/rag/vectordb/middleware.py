"""
Glue code that ties together:
- PdfManager -> extracts images
- ColPaliManager -> embeds images & queries
- KGManager -> retrieve relevant information from Knowledge Graph
- QdrantManager -> stores & searches vectors
"""

import hashlib
from pathlib import Path
from typing import List, Any

from src.rag.utils.pdf_manager import PdfManager
from src.rag.embedders.colpali_manager import ColPaliManager
from src.rag.knowledge_graph.knowledge_graph_manager import KGManager
from src.rag.vectordb.qdrant_manager import QdrantManager
from src.rag.rag import Rag
from src.rag.config import load_config

# Load config
cfg = load_config()

class Middleware:
    """Main entry-point for index() and search() used by the app"""
    
    def __init__(self, user_id: str, model_name: str, quantized: bool, create_collection: bool = True):
        # Init manager
        self.pdf_manager = PdfManager()
        # self.colpali_manager = ColPaliManager()
        self.colpali_manager = ColPaliManager(model_name=model_name, quantized=quantized)
        self.kg_manager = KGManager("data/processed/knowledge graph of DDXPlus.xlsx", 'data/processed')
        
        # Create Qdrant fodler
        # qdrant_folder = Path(cfg["paths"]["db_path"])
        # qdrant_folder.mkdir(exist_ok=True)
        
        # Start Qdrant wrapper
        self.db = QdrantManager(
            collection_name=cfg["vector_db"]["collection_name"],
            vector_size=cfg["vector_db"]["vector_size"],
            create_collection=create_collection,
        )
        # Initilize LLM backbone
        self.rag = Rag()
        
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
                "filepath": image_paths[i],
                "original_file": pdf_path
            }
            for i in range(len(image_paths))
        ]
        print(f"Inserting {len(db_payload)} vectors into Qdrant…")
        self.db.insert_images_data(db_payload)
        print("[Middleware] Indexing complete ✓\n")
        return image_paths
    
    def search(
        self, 
        query: str, 
        top_k: int = cfg["retrieval_top_k"]
    ) -> List[Any]:
        """
        Search and retreive relevant document and information
        from vector database and knowledge graph.
        """    
        print(f"[Middleware] Received query/queries")
        # Retrieve from vector database
        results_vectordb = self._search_vectordb(query=query, top_k=top_k)
        # Retrieve from knowledge graph


        return results_vectordb
    
    def _search_vectordb(
            self,
            query: str,
            top_k: int
    ) -> List[Any]:
        """
        Embed each query with ColPali's text encoder, 
        then fetch top-k image pages from vector database.
        """   
        print(f"Searching in vector database")
        results = []

        print(f"Query: {query!r}")
        query_vec = self.colpali_manager.process_text(query)[0]
        result = self.db.search(query_vec, top_k)
        print(f"Relevant document from vector database: {result}")
        results.append(result)
            
        return results
    