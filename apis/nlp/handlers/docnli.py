from .base import BaseHandler
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.models.roberta.modeling_roberta import RobertaModel
import numpy as np

class DocNLIHandler(BaseHandler):
    def __init__(self, config):
        super().__init__(config)
        
    def load_service(self):
        model_ckpt_path = self.config.get('model_ckpt_path')
        pretrain_model_dir = self.config.get('pretrain_model_dir')#名字，及上层config可以改?
        label_list = ["entailment", "not_entailment"]  # , "contradiction"]
        
        self.service = {
            "tokenizer": AutoTokenizer.from_pretrained(pretrain_model_dir),
            "model": RobertaForSequenceClassification(
                pretrain_model_dir, 
                bert_hidden_dim = 1024, 
                tagset_size = len(label_list) # it is the number of labels
            ).to(self.device)
        }
        
        checkpoint = torch.load(model_ckpt_path, map_location=self.device)
        self.service["model"].load_state_dict(checkpoint, strict=False)# GPT推荐的strict=False
        self.service["model"].eval()
        print("DEBUG: Model loaded on", self.device) # GPT推荐的
        
    def _formatter(self, batch):
        return batch
    
    def entailment_score(self, text1, text2):
        tokenizer, model = self.service["tokenizer"], self.service["model"]
        
        encoded_ctx = tokenizer.encode(text1)[:-1]          # remove [SEP] # 这什么名，改一下吧
        encoded_correction = tokenizer.encode(text2)[1:]    # remove [CLS]
        encoded_ctx_truncated = encoded_ctx[:512 - 1 - len(encoded_correction)]  # - [SEP] - encoded_correction
        
        input_ids = torch.LongTensor(encoded_ctx_truncated + [tokenizer.sep_token_id] + encoded_correction).unsqueeze(0).to(self.device)
        # attention_mask = torch.LongTensor([1] * len(input_ids)).unsqueeze(0).to(self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long) # gpt给的建议
        inputs = {'input_ids': input_ids, 'input_mask': attention_mask}
        
        with torch.no_grad():
            # self.model.eval()
            # logits = self.model(**inputs)
            logits = model(**inputs)
            probs = torch.nn.Softmax(dim=1)(logits)
            correct_prob = probs[0][0].item()
        return correct_prob
    
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    def _process_logic(self, formatted_batch):
        results = []
        for sample in formatted_batch:
            context, candidates = sample['context'], sample['candidates'] # 名字和上层输入不对应
            entailment_scores = [self.entailment_score(context, candidate) for candidate in candidates]
            
            ranked_indices = np.argsort(-np.array(entailment_scores)).tolist()[:5] # 取前5个最大值的索引
            results.append([
                {"id": i, "sentence": candidates[i], "entailment": format(entailment_scores[i], ".2f")}
                for i in ranked_indices
            ])
        return results
        # #双层列表，第一层是batch里的samples，最内层是score

'''
From https://github.com/salesforce/DocNLI/blob/main/Code/DocNLI/test_on_docNLI_RoBERTa.py
'''
from transformers.models.roberta.modeling_roberta import RobertaModel
class RobertaForSequenceClassification(nn.Module):
    def __init__(self, pretrain_model_dir, bert_hidden_dim, tagset_size):
        super(RobertaForSequenceClassification, self).__init__()
        self.tagset_size = tagset_size

        self.roberta_single= RobertaModel.from_pretrained(pretrain_model_dir)
        self.single_hidden2tag = RobertaClassificationHead(bert_hidden_dim, tagset_size)

    def forward(self, input_ids, input_mask):
        outputs_single = self.roberta_single(input_ids, input_mask, None)
        hidden_states_single = outputs_single[1]#torch.tanh(self.hidden_layer_2(torch.tanh(self.hidden_layer_1(outputs_single[1])))) #(batch, hidden)

        score_single = self.single_hidden2tag(hidden_states_single) #(batch, tag_set)
        return score_single


class RobertaClassificationHead(nn.Module):
    """wenpeng overwrite it so to accept matrix as input"""

    def __init__(self, bert_hidden_dim, num_labels):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(bert_hidden_dim, num_labels)

    def forward(self, features):
        x = features#[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x