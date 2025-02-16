import stanza
from .base import BaseHandler
from typing import List

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
    
class StanzaNERHandler(BaseHandler):
    def __init__(self, config):
        super().__init__(config)
    
    def load_service(self):
        self.service = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
        
    def _formatter(self, batch) -> List[List[str]]:
        return [i["sentences"] for i in batch]
        
    def _process_logic(self, formatted_batch: List[List[str]]): # -> List[List[List[str]]]:
        '''
        Get a batch (list of lists of sentences) from the formatted batch
        for each sentence,  get the noun chunks and entities as a list
        return the list of lists of strings 
        '''
        batch_results = []
        for sample in formatted_batch:
            sample_result = []
            for sentence in sample:
                result = []
                stanza_doc = self.service(sentence)
                for sent in stanza_doc.sentences:
                    result.extend(get_phrases(sent.constituency, 'NP'))
                    result.extend(get_phrases(sent.constituency, 'VP'))
                    result.extend([word.text for word in sent.words if word.upos in ['NOUN', 'PROPN', 'ADJ', 'ADV']])
                sample_result.append(result)
            batch_results.append(sample_result)
        return batch_results         
    
