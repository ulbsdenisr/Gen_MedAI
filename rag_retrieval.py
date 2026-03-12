import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer


class DiseaseRAGResponder:
    def __init__(self, index_path, metadata_path, model_name="all-MiniLM-L6-v2"):
        # Load embedding model
        self.model = SentenceTransformer(model_name)

        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load metadata
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

    def retrieve_disease(self, disease_name, top_k=1):
        """
        Retrieves the closest disease entry from FAISS.
        """
        query_vector = self.model.encode([disease_name]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)

        idx = indices[0][0]
        return self.metadata[idx]

    def build_reply(self, disease_list):
        """
        Receives a list of disease names and builds a user-facing reply.
        """
        retrieved = []

        for disease in disease_list:
            entry = self.retrieve_disease(disease)
            retrieved.append(entry)

        # Build response text
        response = """Based on your described symptoms, you may be suffering of:\n\n"""

        for entry in retrieved:
            response += f"{entry['disease'].title()}\n"
            response += "Common associated symptoms:\n"

            for symptom in entry["symptoms"]:
                response += f"- {symptom}\n"

            response += "\n"

        return response