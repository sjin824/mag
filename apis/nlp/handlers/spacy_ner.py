import spacy
from .base import BaseHandler

class SpacyNERHandler(BaseHandler):
    def __init__(self, config):
        super().__init__(config)
        
    def load_service(self):
        self.service = spacy.load('en_core_web_lg')
        
    def _formatter(self, batch: dict):
        return batch
    
    def _process_logic(self, formatted_batch):
        results = []
        for spacy_doc in self.service.pipe(formatted_batch):
            result = []
            result += [noun_chunk.text for noun_chunk in spacy_doc.noun_chunks]
            result += [entity.text for entity in spacy_doc.ents]
            results.append(result)
        return results