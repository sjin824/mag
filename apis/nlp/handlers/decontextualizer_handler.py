from os import path
import tensorflow as tf
import tensorflow_text

MODEL_SIZE = "base" #@param["base", "3B", "11B"]

# Download the model from: https://console.cloud.google.com/storage/browser/decontext_dataset
# Replaced the DATASET_BUCKET with the actual path
# DATASET_BUCKET = 'gs://decontext_dataset'
DATASET_BUCKET = '/home/sjin824/pyprojects/mag/decontextualization'

SAVED_MODELS = {
    "base": f'{DATASET_BUCKET}/t5_base/1611267950',
    "3B": f'{DATASET_BUCKET}/t5_3B/1611333896',
    "11B": f'{DATASET_BUCKET}/t5_11B/1605298402'
}

SAVED_MODEL_PATH = SAVED_MODELS[MODEL_SIZE]
DEV = path.join(DATASET_BUCKET, 'decontext_dev.jsonl')
SAVED_MODEL_PATH = path.join(DATASET_BUCKET, 't5_base/1611267950')

def load_predict_fn(model_path):
    print("Loading SavedModel in eager mode.")
    imported = tf.saved_model.load(model_path, ["serve"])
    return lambda x: imported.signatures['serving_default'](
        tf.constant(x))['outputs'].numpy()

predict_fn = load_predict_fn(SAVED_MODEL_PATH)

def decontextualize(input):
    return predict_fn([input])[0].decode('utf-8')

def create_input(paragraph,
                 target_sentence_idx,
                 page_title='',
                 section_title=''):
    """Creates a single Decontextualization example input for T5.

    Args:
      paragraph: List of strings. Each string is a single sentence.
      target_sentence_idx: Integer index into `paragraph` indicating which
        sentence should be decontextualized.
      page_title: Optional title string. Usually Wikipedia page title.
      section_title: Optional title of section within page.
    """
    prefix = ' '.join(paragraph[:target_sentence_idx])
    target = paragraph[target_sentence_idx]
    suffix = ' '.join(paragraph[target_sentence_idx + 1:])
    return ' [SEP] '.join((page_title, section_title, prefix, target, suffix))

# ================ All above is from https://github.com/google-research/language/blob/master/language/decontext/decontextualization_demo.ipynb ================
paragraph = [
  "Gagarin was a keen sportsman and played ice hockey as a goalkeeper.",
  "He was also a basketball fan and coached the Saratov Industrial Technical School team, as well as being a referee.",
  "In 1957, while a cadet in flight school, Gagarin met Valentina Goryacheva at the May Day celebrations at the Red Square in Moscow.",
  "She was a medical technician who had graduated from Orenburg Medical School.",
  "They were married on 7 November of the same year, the same day Gagarin graduated from his flight school, and they had two daughters.",
  "Yelena Yurievna Gagarina, born 1959, is an art historian who has worked as the director-general of the Moscow Kremlin Museums since 2001; and Galina Yurievna Gagarina, born 1961, is a professor of economics and the department chair at Plekhanov Russian University of Economics in Moscow."
]

page_title = 'Yuri Gagarin'
section_title = 'Personal Life'  # can be empty
target_sentence_idx = 4  # zero-based index
d = decontextualize(
        create_input(paragraph, target_sentence_idx, page_title,
                     section_title))
print(f'Original sentence:         {paragraph[target_sentence_idx]}\n'
      f'Decontextualized sentence: {d}')