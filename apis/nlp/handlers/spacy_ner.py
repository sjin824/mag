import spacy
from .base import BaseHandler
from typing import List

class SpacyNERHandler(BaseHandler):
    def __init__(self, config):
        super().__init__(config)
        
    def load_service(self):
        self.service = spacy.load('en_core_web_lg')
        
    def _formatter(self, batch) -> List[List[str]]:
        return [i["sentences"] for i in batch]
    
    def _process_logic(self, formatted_batch: List[List[str]]): # -> List[List[str]]:
        '''
        Get a batch (list of lists of sentences) from the formatted batch
        for each sentence, get the noun chunks and entities as a list
        return the list of lists of strings
        '''
        batch_results = []
        for sample in formatted_batch:  
            sample_result = []
            spacy_docs = self.service.pipe(sample)
            for spacy_doc in spacy_docs:
                result = [noun_chunk.text for noun_chunk in spacy_doc.noun_chunks]
                result += [entity.text for entity in spacy_doc.ents]
                sample_result.append(result)
            batch_results.append(sample_result)
        return batch_results