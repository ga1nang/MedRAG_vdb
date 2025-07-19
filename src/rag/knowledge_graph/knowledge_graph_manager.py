import pandas as pd
from utils.utils import preprocess_text
class KGManager:
    def __init__(self, knowledge_gpaph_path):
        self.kg_data = pd.read_excel(knowledge_gpaph_path, usecols=['subject', 'relation', 'object'])

        self.knowledge_graph = {}
        for index, row in self.kg_data.iterrows():
            subject = row['subject']
            relation = row['relation']
            obj = row['object']

            # Bidirectional relationship: subject → object and object → subject
            self.knowledge_graph.setdefault(subject, []).append((relation, obj))
            self.knowledge_graph.setdefault(obj, []).append((relation, subject))

        # Preprocess 'object' column for similarity matching (but skip 'is_a' relations)
        self.kg_data['object_preprocessed'] = self.kg_data.apply(
            lambda row: preprocess_text(row['object']) if row['relation'] != 'is_a' else None,
            axis=1
        )
        self.symptom_nodes = self.kg_data['object_preprocessed'].dropna().unique().tolist()