from os import path
import tensorflow as tf
import tensorflow_text

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

## Loading SavedModel in eager mode
def load_predict_fn(model_path): 
    imported = tf.saved_model.load(model_path, ["serve"])
    return lambda x: imported.signatures['serving_default'](
        tf.constant(x))['outputs'].numpy()
    
SAVED_MODEL_PATH = '/app/tf_models/t5_base/1611267950'
predict_fn = load_predict_fn(SAVED_MODEL_PATH)

# Adapt the input to the model
def create_input(paragraph,
                 target_sentence_idx,
                 page_title='',
                 section_title=''):
    prefix = ' '.join(paragraph[:target_sentence_idx])
    target = paragraph[target_sentence_idx]
    suffix = ' '.join(paragraph[target_sentence_idx + 1:])
    return ' [SEP] '.join((page_title, section_title, prefix, target, suffix))

# Main functionality: decontextualization
def decontextualize(batch_inputs):
    results = []
    for _input in batch_inputs:
        created_input = create_input(
            _input['paragraph'], 
            _input['target_sentence_idx'],
            _input.get('page_title', ''),
            _input.get('section_title', '')
        )
        results.append(predict_fn([created_input])[0].decode('utf-8'))
    return results
