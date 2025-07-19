import os
import pandas as pd
import networkx as nx
from utils.utils import preprocess_text
from embedders.bert_manager import BERTManger

class KGManager:
    def __init__(self, knowledge_graph_path: str, embedding_save_dir: str = "data/processed/symtom_embeddings"):
        """
        Knowledge Graph Manager for loading, processing, and querying the medical KG.

        Args:
            knowledge_graph_path (str): Path to the Excel file containing ['subject', 'relation', 'object'] columns.
            embedding_save_dir (str): Directory to store or load symptom embeddings.
        """
        self.kg_path = knowledge_graph_path
        self.embedding_save_dir = embedding_save_dir
        self.bert_manager = BERTManger()

        # Load KG data
        self.kg_data = pd.read_excel(self.kg_path, usecols=['subject', 'relation', 'object'])
        self.knowledge_graph = self._build_knowledge_dict()
        self.G = self._build_graph()

        # Preprocess symptom nodes (skip hierarchical 'is_a' links)
        self.symptom_nodes = self._extract_symptom_nodes()
        self.symptom_embeddings = self._load_or_generate_embeddings()

        # Predefined categories (could also be loaded from a config file)
        self.categories = [
            "cardiovascular_system", "respiratory_system", "gastrointestinal_system",
            "neurological_and_muscular_system", "infectious_diseases",
            "autoimmune_and_immunological_diseases", "hematological_disorders",
            "trauma_and_injury_related_conditions", "psychiatric_and_stress_related_disorders"
        ]

    # ---------- Internal Helpers ----------

    def _build_knowledge_dict(self) -> dict:
        """Construct adjacency list representation of the KG (without redundant bidirectional links)."""
        kg = {}
        for _, row in self.kg_data.iterrows():
            kg.setdefault(row['subject'], []).append((row['relation'], row['object']))
        return kg

    def _build_graph(self) -> nx.Graph:
        """Builds an undirected NetworkX graph from the adjacency list."""
        G = nx.Graph()
        for node, edges in self.knowledge_graph.items():
            for relation, neighbor in edges:
                G.add_edge(node, neighbor, relation=relation)
        return G

    def _extract_symptom_nodes(self):
        """Extract preprocessed symptom nodes for embedding generation."""
        self.kg_data['object_preprocessed'] = self.kg_data.apply(
            lambda row: preprocess_text(row['object']) if row['relation'] != 'is_a' else None,
            axis=1
        )
        return self.kg_data['object_preprocessed'].dropna().unique().tolist()

    def _load_or_generate_embeddings(self) -> dict:
        """Loads precomputed embeddings if available, otherwise generates and saves them."""
        os.makedirs(self.embedding_save_dir, exist_ok=True)
        save_path = os.path.join(self.embedding_save_dir, "symptom_embeddings")

        if os.path.exists(save_path):
            return self.bert_manager.load_embeddings(save_path)

        embeddings = self.bert_manager.get_symptom_embeddings(
            symptom_nodes=self.symptom_nodes,
            save_path=save_path
        )
        return {node: emb for node, emb in zip(self.symptom_nodes, embeddings)}