from nltk.tokenize import sent_tokenize
from simcse import SimCSE
import numpy as np
from typing import List

model = SimCSE("princeton-nlp/sup-simcse-roberta-large")

def sentence_tokenize(original_fulltext_batch: List[str]) -> List[List[str]]:
    """Tokenize input texts into sentences using NLTK."""
    return [sent_tokenize(text) for text in original_fulltext_batch]

def rank_sentences(tokenized_sentence_batch: List[List[str]], original_fulltext_batch: List[str]) -> List[List[dict]]:
    """Rank sentences based on similarity to the fulltext using SimCSE."""
    result = []
    for tokenized_sentences, original_text in zip(tokenized_sentence_batch, original_fulltext_batch):
        sim_scores = model.similarity(tokenized_sentences, [original_text])
        ranked_indices = np.argsort(-sim_scores[:, 0]).tolist()
        result.append([
            {"id": i, "sentence": tokenized_sentences[i], "similarity_score": float(sim_scores[i, 0])}
            for i in ranked_indices
        ])
    return result
