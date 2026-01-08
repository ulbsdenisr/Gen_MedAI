"""
Utility functions for normalizing and splitting symptom entities
"""
import re

SPLIT_PATTERN = r",| and | with | plus | accompanied by "

def normalize_and_split_symptoms(entities):
    """
    Split compound symptom strings and normalize them.
    
    Args:
        entities: List of symptom strings
        
    Returns:
        List of normalized, individual symptoms with duplicates removed
    """
    results = []
    for ent in entities:
        parts = re.split(SPLIT_PATTERN, ent)
        for p in parts:
            p = p.strip()
            if len(p) > 2:
                results.append(p)
    return list(dict.fromkeys(results))  # remove duplicates, keep order