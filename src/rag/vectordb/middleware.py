"""
Glue code that ties together:
- PdfManager -> extracts images
- ColPaliManager -> embeds images & queries
- KGManager -> retrieve relevant information from Knowledge Graph
- QdrantManager -> stores & searches vectors
"""
import fitz
from pathlib import Path
from typing import List, Any

from src.rag.utils.pdf_manager import PdfManager
from src.rag.utils.utils import extract_text_from_pdf, truncate_text
from src.rag.embedders.colpali_manager import ColPaliManager
from src.rag.re_ranker.re_ranker_manager import ReRankerManager
from src.rag.knowledge_graph.knowledge_graph_manager import KGManager
from src.rag.vectordb.qdrant_manager import QdrantManager
from src.rag.rag import Rag
from src.rag.config import load_config

# Load config
cfg = load_config()

class Middleware:
    """Main entry-point for index() and search() used by the app"""
    
    def __init__(self, kg_path: str, model_name: str, quantized: bool, create_collection: bool = True, enable_rag: bool = False, quantize_llm: bool = False, quantization_type_llm: str = "4bit"):
        # Init manager
        self.pdf_manager = PdfManager()
        # self.colpali_manager = ColPaliManager()
        self.colpali_manager = ColPaliManager(model_name=model_name, quantized=quantized)
        self.re_ranker_manager = ReRankerManager()
        self.kg_manager = KGManager(kg_path, 'data/processed')
        
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
        if enable_rag:
            if quantize_llm:
                self.rag = Rag(quantize=quantize_llm, quantization_type=quantization_type_llm)
            else:
                self.rag = Rag(quantize=quantize_llm)
        
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
        print(f"Query: {query!r}")
        # Retrieve from vector database
        results_vectordb = self._search_vectordb(query=query, top_k=top_k)
        # Retrieve from knowledge graph
        results_kg, _, _ = self._search_knowledge_graph(query=query, top_k=1)

        return results_vectordb, results_kg
    
    def _search_vectordb(
            self,
            query: str,
            top_k: int
    ) -> List[Any]:
        """
        Embed each query with ColPali's text encoder, 
        then fetch top-k image pages from vector database.
        """ 
        print("--------------------------------------------------------------------------------")
        print(f"Searching in vector database")

        query_vec = self.colpali_manager.process_text(query)[0]
        print(top_k)
        result = self.db.search(query_vec, top_k)
        print(f"Relevant document from vector database: {len(result)} files\n{result}")
        result = self.re_ranker_manager.re_rank(query=query, vectordb_docs=result)
        print("--------------------------------------------------------------------------------")
        return result
    
    def _search_knowledge_graph(
            self,
            query: str,
            top_k: int,
    ) -> str:
        """
        Retrive relevant information from Knowledge Graph
        """
        print("--------------------------------------------------------------------------------")
        print("Searching in Knowledge Graph")
        # Feature decomposition the query
        features = self.rag.feature_decomposition(query=query)
        histories = features["history"]
        symptoms = features["symptoms"]
        # Retrieve relevant info from KG
        results = self.kg_manager.get_additional_info_from_level_2(
            histories=histories,
            symtoms=symptoms,
            top_n_categries=top_k,
            top_n_symptoms=top_k,
        )
        # print(f"Relevant information from Knowledge Graph: \n{results}")
        print("--------------------------------------------------------------------------------")
        return results, histories, symptoms
    
    def get_answer_from_medgemma(
        self,
        query: str,
        images_path: List[str],
        retrieved_docs: List[str],
        kg_info: str
    ) -> str:
        # Limit number of retrieved docs
        relevant_docs = ["# Retrieved Clinical Cases from Vector Database"]
        for i, doc in enumerate(retrieved_docs):  # max 3 docs
            snippet = extract_text_from_pdf(doc, max_pages=4, max_chars=8000)
            relevant_docs.append(f"Case {i+1}:\n{snippet}")

        docs_block = "\n".join(relevant_docs)
        kg_block = truncate_text(kg_info, max_chars=2000)

        fused_query = (
            f"{query}.\nThese are relevant documents retrieved from the vector database:\n"
            f"{docs_block}\n\n"
            f"And these are relevant information retrieved from the knowledge graph:\n"
            f"{kg_block}"
        )
        return self.rag.get_answer_from_medgemma(fused_query, images_path)
