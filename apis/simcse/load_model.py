from simcse import SimCSE

def preload_models(model_name):
    try:
        SimCSE(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to preload SimCSE model: {e}")

if __name__ == "__main__":
    preload_models("princeton-nlp/sup-simcse-roberta-large")
