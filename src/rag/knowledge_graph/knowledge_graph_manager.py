import os
import pandas as pd
import networkx as nx
from typing import List, Any
from pandas import DataFrame
from sklearn.metrics.pairwise import cosine_similarity
from src.rag.utils.utils import preprocess_text
from src.rag.embedders.bert_manager import BERTManger

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
    
    def _load_level_3_to_level_2(self):
        return {
            # Cardiovascular System
            "atrial_fibrillation": "cardiovascular_system",
            "myocarditis": "cardiovascular_system",
            "pericarditis": "cardiovascular_system",
            "psvt": "cardiovascular_system",
            "possible_nstemi_stemi": "cardiovascular_system",
            "stable_angina": "cardiovascular_system",
            "unstable_angina": "cardiovascular_system",
            "pulmonary_embolism": "cardiovascular_system",
            # Respiratory System
            "acute_copd_exacerbation_infection": "respiratory_system",
            "bronchiectasis": "respiratory_system",
            "bronchiolitis": "respiratory_system",
            "bronchitis": "respiratory_system",
            "bronchospasm_acute_asthma_exacerbation": "respiratory_system",
            "pulmonary_embolism": "respiratory_system",
            "pulmonary_neoplasm": "respiratory_system",
            "spontaneous_pneumothorax": "respiratory_system",
            "urti": "respiratory_system",
            "viral_pharyngitis": "respiratory_system",
            "whooping_cough": "respiratory_system",
            "acute_laryngitis": "respiratory_system",
            "acute_pulmonary_edema": "respiratory_system",
            "croup": "respiratory_system",
            "larygospasm": "respiratory_system",
            "epiglottitis": "respiratory_system",
            "pneumonia": "respiratory_system",
            # Gastrointestinal System
            "gerd": "gastrointestinal_system",
            "boerhaave_syndrome": "gastrointestinal_system",
            "pancreatic_neoplasm": "gastrointestinal_system",
            "scombroid_food_poisoning": "gastrointestinal_system",
            "inguinal_hernia": "gastrointestinal_system",
            # Neurological and Muscular System
            "myasthenia_gravis": "neurological_and_muscular_system",
            "guillain_barre_syndrome": "neurological_and_muscular_system",
            "cluster_headache": "neurological_and_muscular_system",
            "acute_dystonic_reactions": "neurological_and_muscular_system",
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
            # Autoimmune and Immunological Diseases
            "sle": "autoimmune_and_immunological_diseases",
            "sarcoidosis": "autoimmune_and_immunological_diseases",
            "anaphylaxis": "autoimmune_and_immunological_diseases",
            "allergic_sinusitis": "autoimmune_and_immunological_diseases",
            # Hematological Disorders
            "hematological_disorders": "hematological_disorders",
            # Trauma and Injury Related Conditions
            "spontaneous_rib_fracture": "trauma_and_injury_related_conditions",
            "spontaneous_pneumothorax": "trauma_and_injury_related_conditions",
            "inguinal_hernia": "trauma_and_injury_related_conditions",
            # Psychiatric and Stress Related Disorders
            "panic_attack": "psychiatric_and_stress_related_disorders",
        }

    
    # ---------- Public Query APIs ----------
    def get_additional_info_from_level_2(self, kg_path, histories, symtoms, top_n_categries, top_n_symptoms):
        """
        Given a participant ID, this function identifies the most relevant Level 2 category (eL2_s)
        and retrieves knowledge graph triples (eL3, relation, eL4) associated with that category.
        These triples are then formatted as strings for use in LLM reasoning prompts.

        Parameters:
        - participant_no (int): ID of the participant/patient
        - kg_path (str): Path to the knowledge graph Excel file
        - top_n (int): Number of top categories to retrieve
        - match_n (int): Number of features to match during upward traversal

        Returns:
        - str: A string of triples representing relevant knowledge from the KG
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
            additional_info = []
            for (subject, relation), objects in merged_info.items():
                sentence = f"{subject} {relation} {', '.join(objects)}"
                additional_info.append(sentence)

        # Return final info string if available
        if not additional_info:
            print("No additional information found.")
            return None

        final_info = ', '.join(additional_info)
        print("Additional Info:", final_info)
        return final_info

    def main_get_category_and_level3(self, histories: DataFrame, symptoms: DataFrame, top_n_symptoms: int = 5, top_n_categries: int = 5) -> List:
        """Main function to return top-n categories for a given participant case."""

        # Extract patient input fields
        disease_histories = histories
        disease_symptoms = symptoms

        print(f'disease_histories: {disease_histories}')
        print(f'disease_symptoms: {disease_symptoms}')

        # Handle missing values
        disease_histories = '' if pd.isna(disease_histories) else disease_histories
        disease_symptoms = '' if pd.isna(disease_symptoms) else disease_symptoms

        # Get top-n symptoms based on embeddings
        def process_symptom_field(field_value, symptom_nodes, symptom_embeddings, n):
            return self.find_top_n_similar_symptoms(field_value, symptom_nodes, symptom_embeddings, n) if field_value else []

        top_5_history_nodes = process_symptom_field(disease_histories, self.symptom_nodes, self.symptom_embeddings, top_n_symptoms)
        top_5_symptom_nodes = process_symptom_field(disease_symptoms, self.symptom_nodes, self.symptom_embeddings, top_n_symptoms)

        # Map back to original terms for interpretability
        top_5_history_nodes_original = self.kg_data.loc[self.kg_data['object_preprocessed'].isin(top_5_history_nodes), 'object'].drop_duplicates()
        top_5_symptom_nodes_original = self.kg_data.loc[self.kg_data['object_preprocessed'].isin(top_5_symptom_nodes), 'object'].drop_duplicates()

        # Use all matched symptoms to vote on closest category
        most_similar_category = self.find_closest_category(
            list(top_5_history_nodes_original) +
            list(top_5_symptom_nodes_original) +
            self.categories,
            top_n_categries
        )
        return most_similar_category
    
    def find_top_n_similar_symptoms(self, query, symptom_nodes, symptom_embeddings, n):
        """Find top-N similar symptoms from query using cosine similarity."""
        if pd.isna(query) or not query:
            return []

        # Preprocess and embed the query
        query_preprocessed = preprocess_text(query)
        query_embedding = self.bert_manager.generate_embedding(query_preprocessed)
        if not query_embedding:
            return []

        # Truncate in case of mismatch between embedding and symptom count
        if len(symptom_embeddings) > len(symptom_nodes):
            symptom_embeddings = symptom_embeddings[:len(symptom_nodes)]

        # Calculate cosine similarity
        similarities = cosine_similarity([query_embedding], symptom_embeddings).flatten()

        # Select top-n unique symptoms with threshold > 0.5
        top_n_symptoms = []
        unique_symptoms = set()
        top_n_indices = similarities.argsort()[::-1]

        for i in top_n_indices:
            if similarities[i] > 0.5 and symptom_nodes[i] not in unique_symptoms:
                top_n_symptoms.append(symptom_nodes[i])
                unique_symptoms.add(symptom_nodes[i])
            if len(top_n_symptoms) == n:
                break

        return top_n_symptoms
