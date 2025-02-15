import tensorflow as tf
import tensorflow_text
from .base import BaseHandler

''' 
================ The codes below are partially from: ================
https://github.com/google-research/language/blob/master/language/decontext/decontextualization_demo.ipynb 
================ Download the model from: ================
https://console.cloud.google.com/storage/browser/decontext_dataset
store the models in apis/nlp/tf_models
Currently, only the model: t5_base is available

Doubt: the model doesn't run on GPU
Improve: clean cache, develop a interface to deload the model
'''
class DecontextualizerHandler(BaseHandler):
    def __init__(self, config):
        super().__init__(config)
        
    def load_service(self):
        model_path = self.config['model_path']
        self.service = tf.saved_model.load(model_path, ["serve"])

    def _formatter(self, batch: dict):
        formatted_batch = []
        for sample in batch:
            paragraph = sample['paragraph']
            target_sentence_idx = sample['target_sentence_idx']
            page_title = sample.get('page_title', '')
            section_title = sample.get('section_title', '')
            
            prefix = ' '.join(paragraph[:target_sentence_idx])
            target = paragraph[target_sentence_idx]
            suffix = ' '.join(paragraph[target_sentence_idx + 1:])
            result = ' [SEP] '.join((page_title, section_title, prefix, target, suffix))
            formatted_batch.append(result)
        return formatted_batch
    
    # Decontextualization.
    def _process_logic(self, formatted_batch):
        results = []
        for sample in formatted_batch:
            results.append(self.service.signatures['serving_default'](
                tf.constant([sample]))['outputs'].numpy()[0].decode('utf-8')) 
        return results

    
