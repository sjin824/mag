import spacy
import stanza
from simcse import SimCSE
import time
import argparse

def load_with_retry(load_func, model_name, retries=3, delay=5):
    """Automatically retry model loading and track execution time."""
    for attempt in range(retries):
        try:
            start = time.time()
            result = load_func()
            print(f"[{model_name}] loaded successfully in {time.time() - start:.2f} seconds")
            return result
        except Exception as e:
            print(f"[{model_name}] failed on attempt {attempt + 1}: {e}")
            # if attempt < retries - 1:
            #     print(f"Waiting {delay} seconds before retrying...")
            #     time.sleep(delay)
    raise RuntimeError(f"[{model_name}] failed to load after {retries} attempts.")

def main(spacy_model, stanza_model, simcse_model):
    if spacy_model:
        load_with_retry(lambda: spacy.load(spacy_model), f"spaCy {spacy_model}")
    if stanza_model:
        load_with_retry(lambda: stanza.Pipeline(lang='en', processors=stanza_model, download_method=None), f"Stanza {stanza_model}")
    if simcse_model:
        load_with_retry(lambda: SimCSE(simcse_model), f"SimCSE {simcse_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamically load NLP models.")
    parser.add_argument("--spacy", type=str, help="Specify the spaCy model, e.g., en_core_web_lg")
    parser.add_argument("--stanza", type=str, help="Specify the Stanza processors, e.g., tokenize,pos,constituency")
    parser.add_argument("--simcse", type=str, help="Specify the SimCSE model, e.g., princeton-nlp/sup-simcse-roberta-large")
    args = parser.parse_args()

    main(
        spacy_model=args.spacy, 
        stanza_model=args.stanza, 
        simcse_model=args.simcse
    )