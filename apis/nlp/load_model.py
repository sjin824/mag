import spacy
import stanza

def preload_models():
    try:
        spacy.load('en_core_web_lg')
        stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    except Exception as e:
        raise RuntimeError(f"Failed to preload nlp tools: {e}")

if __name__ == "__main__":
    preload_models()
