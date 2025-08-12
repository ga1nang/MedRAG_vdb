import os
import pandas as pd
import networkx as nx
import numpy as np
from typing import List, Any
from pandas import DataFrame
from sklearn.metrics.pairwise import cosine_similarity
from src.rag.utils.utils import preprocess_text
from src.rag.embedders.bert_manager import BERTManger

class KGManager:
    def __init__(self, knowledge_graph_path: str, embedding_save_dir: str = "data/processed"):
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
            "neurological_and_muscular_system", "infectious_diseases"
        ]

        # Predefined the diseases im each categories
        self.level_3_to_level_2 = self._load_level_3_to_level_2()

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

    def _load_or_generate_embeddings(self) -> dict[str, np.ndarray]:
        os.makedirs(self.embedding_save_dir, exist_ok=True)
        path = os.path.join(self.embedding_save_dir, "symptom_embeddings.npy")

        if os.path.exists(path):
            arr = np.load(path, allow_pickle=True)
            return dict(zip(self.symptom_nodes, arr))

        vecs = [
            self.bert_manager.generate_embedding(node).squeeze(0).cpu().numpy()
            for node in self.symptom_nodes
        ]
        np.save(path, np.array(vecs, dtype=object))       
        return dict(zip(self.symptom_nodes, vecs))
    
    def _load_level_3_to_level_2(self):
        return {
            # Cardiovascular System
            "myocarditis": "cardiovascular_system",
            "pericarditis": "cardiovascular_system",
            # Respiratory System
            "acute_copd_exacerbation_infection": "respiratory_system",
            "bronchiectasis": "respiratory_system",
            "bronchiolitis": "respiratory_system",
            "bronchitis": "respiratory_system",
            "bronchospasm_acute_asthma_exacerbation": "respiratory_system",
            "urti": "respiratory_system",
            "viral_pharyngitis": "respiratory_system",
            "whooping_cough": "respiratory_system",
            "acute_laryngitis": "respiratory_system",
            "croup": "respiratory_system",
            "epiglottitis": "respiratory_system",
            "pneumonia": "respiratory_system",
            # Gastrointestinal System
            "scombroid_food_poisoning": "gastrointestinal_system",
            # Neurological and Muscular System
            "guillain_barre_syndrome": "neurological_and_muscular_system",
            # Infectious Diseases
            "tuberculosis": "infectious_diseases",
            "hiv_initial_infection": "infectious_diseases",
            "ebola": "infectious_diseases",
            "influenza": "infectious_diseases",
            "chagas": "infectious_diseases",
            "acute_otitis_media": "infectious_diseases",
            "acute_rhinosinusitis": "infectious_diseases",
            "allergic_sinusitis": "infectious_diseases",
            "chronic_rhinosinusitis": "infectious_diseases",
            "pneumonia": "infectious_diseases",
        }

    
    # ---------- Public Query APIs ----------
    def get_additional_info_from_level_2(self, histories, symtoms, top_n_categries, top_n_symptoms):
        """
        Given a participant ID, this function identifies the most relevant Level 2 category (eL2_s)
        and retrieves knowledge graph triples (eL3, relation, eL4) associated with that category.
        These triples are then formatted as strings for use in LLM reasoning prompts.
        """

        # Run upward traversal to get top-n most relevant Level 2 categories
        level_2_values = self.main_get_category_and_level3(histories, symtoms, top_n_categries, top_n_symptoms)
        additional_info = []

        # For each predicted Level 2 category...
        for level_2_value in level_2_values:
            # Find all Level 3 (disease) nodes that belong to this Level 2 category
            relevant_level_3_descriptions = [
                desc for desc, level2 in self.level_3_to_level_2.items() if level2 == level_2_value
            ]
            print("Relevant Level 3 Descriptions:", relevant_level_3_descriptions)

            # Skip if no level 3 descriptions are associated with this category
            if not relevant_level_3_descriptions:
                print("No Level 3 descriptions found for Level 2:", level_2_value)
                continue

            # Collect triples (subject, relation, object) for each Level 3 disease
            merged_info = {}
            for level_3 in relevant_level_3_descriptions:
                # Find all rows where this level_3 disease is the subject
                related_info = self.kg_data[self.kg_data['subject'] == level_3]

                # Skip if no info found for this disease
                if related_info.empty:
                    print(f"No related information found in KG for: {level_3}")
                else:
                    # Store all (relation, object) pairs under the same subject
                    for _, row in related_info.iterrows():
                        subject = row['subject']
                        relation = row['relation'].replace('_', ' ')  # Make relations more natural-language friendly
                        obj = row['object']

                        # Merge objects under the same (subject, relation) pair
                        if (subject, relation) in merged_info:
                            merged_info[(subject, relation)].append(obj)
                        else:
                            merged_info[(subject, relation)] = [obj]

            # Convert merged triples into readable sentence strings
            for (subject, relation), objects in merged_info.items():
                sentence = f"{subject} {relation} {', '.join(objects)}"
                additional_info.append(sentence)

        # Return final info string if available
        if not additional_info:
            print("No additional information found.")
            return None

        final_info = '\n'.join(additional_info)
        # print("Additional Info:", final_info)
        return final_info

    def main_get_category_and_level3(
        self,
        histories: List[str] | str,
        symptoms: List[str] | str,
        top_n_symptoms: int = 5,
        top_n_categories: int = 5,
    ) -> List[str]:
        # ① Lists ➜ one concatenated string
        if isinstance(histories, (list, tuple, set)):
            disease_histories = " ".join(histories)
        else:
            disease_histories = histories or ""

        if isinstance(symptoms, (list, tuple, set)):
            disease_symptoms = " ".join(symptoms)
        else:
            disease_symptoms = symptoms or ""

        # ② Get top‑N KG symptom nodes
        top_5_history_nodes = self.find_top_n_similar_symptoms(
            disease_histories, self.symptom_nodes, self.symptom_embeddings, 5
        )
        top_5_symptom_nodes = self.find_top_n_similar_symptoms(
            disease_symptoms,  self.symptom_nodes, self.symptom_embeddings, 5
        )

        # Map back to original terms for interpretability
        top_5_history_nodes_original = self.kg_data.loc[self.kg_data['object_preprocessed'].isin(top_5_history_nodes), 'object'].drop_duplicates()
        top_5_symptom_nodes_original = self.kg_data.loc[self.kg_data['object_preprocessed'].isin(top_5_symptom_nodes), 'object'].drop_duplicates()

        # Use all matched symptoms to vote on closest category
        most_similar_category = self.find_closest_category(
            list(top_5_history_nodes_original) +
            list(top_5_symptom_nodes_original),
            self.categories,
            top_n_categories
        )
        return most_similar_category
    
    def find_top_n_similar_symptoms(
            self, query, symptom_nodes, symptom_embeddings, n=5
    ) -> list[str]:
        if not query:
            return []

        # 1. query vector (1, d)
        q_vec = (
            self.bert_manager.generate_embedding(preprocess_text(query))
            .squeeze()              # drops any singleton dims
            .cpu()
            .numpy()
            .reshape(1, -1)
        )

        # 2. symptom matrix (n_symptoms, d)
        sym_matrix = np.vstack(
            [np.asarray(symptom_embeddings[node]).reshape(-1) for node in symptom_nodes]
        )

        # 3. cosine similarity
        sims = cosine_similarity(q_vec, sym_matrix).flatten()

        # 4. pick top‑N ≥ 0.5
        top_idxs = sims.argsort()[::-1]
        chosen, seen = [], set()
        for idx in top_idxs:
            if sims[idx] > 0.5 and symptom_nodes[idx] not in seen:
                chosen.append(symptom_nodes[idx])
                seen.add(symptom_nodes[idx])
                if len(chosen) == n:
                    break
        return chosen
    
    def find_closest_category(self, top_symptoms, categories, top_n):
        """Return top-n closest categories based on symptom-diagnosis-category paths."""

        # Edge case: no symptoms
        if isinstance(top_symptoms, pd.Series) and top_symptoms.empty:
            return None

        # Initialize votes for each category (eL2)
        category_votes = {category: 0 for category in categories}

        # Remove duplicate symptoms
        top_symptoms = list(set(top_symptoms))

        for symptom in top_symptoms:
            # If symptom not in graph, skip
            if symptom not in self.G:
                continue

            # Step 1: Retrieve diseases (eL3) connected to the symptom tᵢ
            diagnosis_nodes = self.get_diagnoses_for_symptom(symptom)

            for diagnosis in diagnosis_nodes:
                for single_diagnosis in diagnosis.split(','):  # handle multiple names
                    single_diagnosis = single_diagnosis.strip().replace(' ', '_').lower()

                    if single_diagnosis not in self.G:
                        continue

                    # Step 2: Find closest category (eL2) for this diagnosis
                    min_distance = float('inf')
                    closest_category = None

                    for category in categories:
                        if category not in self.G:
                            continue

                        try:
                            # Compute shortest path from eL3 to eL2
                            distance = nx.shortest_path_length(self.G, source=single_diagnosis, target=category)
                            if distance < min_distance:
                                min_distance = distance
                                closest_category = category
                        except nx.NetworkXNoPath:
                            continue

                    # Step 3: Vote for closest category
                    if closest_category:
                        category_votes[closest_category] += 1

        # Step 4: Sort categories by number of votes (∑ χ(tᵢ, eL2ⱼ))
        sorted_categories = sorted(category_votes.items(), key=lambda x: x[1], reverse=True)

        # Step 5: Return top-N most voted categories
        top_n_categories = [sorted_categories[i][0] for i in range(top_n)]
        return top_n_categories

    def get_diagnoses_for_symptom(self, symptom):
        """Find all diagnoses (neighbor nodes) connected to a symptom node."""
        diagnoses = []
        if symptom in self.G:
            for neighbor in self.G.neighbors(symptom):
                edge_data = self.G.get_edge_data(neighbor, symptom)
                if edge_data and 'relation' in edge_data and edge_data['relation'] != 'is_a':
                    diagnoses.append(neighbor)
        return diagnoses