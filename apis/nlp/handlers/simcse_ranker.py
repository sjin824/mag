from nltk.tokenize import sent_tokenize
from simcse import SimCSE
import numpy as np
from typing import List
from .base import BaseHandler

class SimcseRankerHandler(BaseHandler):
    def __init__(self, config):
        super().__init__(config)
        
    def load_service(self):
        self.service = SimCSE("princeton-nlp/sup-simcse-roberta-large")
        
    def _formatter(self, batch: dict):
        return batch
    
    def _process_logic(self, formatted_batch):
        results = []
        for sample in formatted_batch:
            original_text = sample['original_text']
            
            tokenized = sent_tokenize(original_text)
            sim_scores = self.service.similarity(tokenized, [original_text])
            sim_scores = np.round(sim_scores, 2) # ?这里不一定对？
            ranked_indices = np.argsort(-sim_scores[:, 0]).tolist()
            result.append([
                {"id": i, "sentence": tokenized_sentences[i], "similarity_score": sim_scores[i, 0]}
                for i in ranked_indices
            ])
        return results
    