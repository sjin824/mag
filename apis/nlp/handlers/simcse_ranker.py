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
        
    def _formatter(self, batch) -> List[str]:
        return [i["full_text"] for i in batch]
    
    def _process_logic(self, formatted_batch: List[str]): # -> List[List[dict]]
        '''
        Get a list of full texts from the formatted batch
        for each full text, tokenize as a list of sentences
        for each sentence, get the similarity score with the full text itself
        return the list of lists of dictionaries, each dictionary stands for a sentence
        '''
        results = []
        for full_text in formatted_batch: # full_text should be a document as a string
            tokenized = sent_tokenize(full_text) # should be a list of strings
            
            sim_scores = self.service.similarity(tokenized, [full_text])
            sim_scores = sim_scores.astype(float).tolist()
            
            ranked_indices = np.argsort(-np.array(sim_scores)).ravel().tolist()
            results.append([
                {"id": i, "sentence": tokenized[i], "similarity": format(sim_scores[i][0], ".2f")}
                for i in ranked_indices
            ])
        return results
    