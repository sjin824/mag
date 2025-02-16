from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .base import BaseHandler

class MixQGHandler(BaseHandler):
    def __init__(self, config):
        super().__init__(config)

    def load_service(self):
        self.service = {
            "tokenizer": AutoTokenizer.from_pretrained('t5-base'),
            "model": AutoModelForSeq2SeqLM.from_pretrained('Salesforce/mixqg-base').to(self.device)
        }

    def _formatter(self, batch):
        formatted_batch = []
        for sample in batch:
            context = sample['context']
            answers = sample['answers'] # A list of entities.

            formatted_sample = []
            for answer in answers:
                formatted_sample.append(f"{answer} \\n {context}")
            formatted_batch.append(formatted_sample)
        return formatted_batch
    
    # Question generation based on 'context' and 'answers'.
    def _process_logic(self, formatted_batch):
        results = []
        tokenizer = self.service["tokenizer"]
        model = self.service["model"]
        
        for sample in formatted_batch:
            input_ids = tokenizer(sample, return_tensors="pt", padding='longest',
                                    truncation=True, max_length=1024).input_ids.to(self.device)
            generated_ids = model.generate(input_ids, max_length=32, num_beams=4)
            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            results.append([output])
        return results

    
