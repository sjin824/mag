import stanza
from .base import BaseHandler

class StanzaNERHandler(BaseHandler):
    def __init__(self, config):
        super().__init__(config)
    
    def load_service(self):
        self.service = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
        
    def _formatter(self, batch: dict):
        return batch
        
    def _process_logic(self, formatted_batch):
        results = []
        for sample in formatted_batch:
            stanza_doc = self.service(sample)
            result = []
            for sent in stanza_doc.sentences:
                result.extend(get_phrases(sent.constituency, 'NP'))
                result.extend(get_phrases(sent.constituency, 'VP'))
                result.extend([word.text for word in sent.words if word.upos in ['NOUN', 'PROPN', 'ADJ', 'ADV']])
            results.append(result)
        return results
    
def get_phrases(tree, label):
    if tree.is_leaf():
        return []
    results = []
    for child in tree.children:
        results += get_phrases(child, label)

    if tree.label == label:
        return [' '.join(tree.leaf_labels())] + results
    else:
        return results