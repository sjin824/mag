from flask import Blueprint, request, jsonify, stream_with_context, Response
from nltk.tokenize import sent_tokenize
from simcse import SimCSE
import numpy as np
import json
from typing import List, Generator
from functools import wraps

simcse_bp = Blueprint("simcse", __name__)
model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
MAX_BATCH_SIZE = 100

# MAX_TEXT_LENGTH = 10000
# def validate_input(text):
#     if len(text) > MAX_TEXT_LENGTH:
#         raise ValueError(f"Input text exceeds maximum length of {MAX_TEXT_LENGTH} characters.")

def handle_api_errors(func):
    """Decorator to handle API errors uniformly."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            return jsonify({"error": f"Unexpected server error: {e}"}), 500
    return wrapper

def validate_request(data):
    """Validate input data for API endpoints."""
    if not data:
        raise ValueError("Request body must be JSON.")
    text_list = data.get("text_list", [])
    batch_size = data.get("batch", 10)
    if not isinstance(text_list, list) or not text_list:
        raise ValueError("Input must be a list of strings, and cannot be empty.")
    if not isinstance(batch_size, int) or batch_size <= 0 or batch_size > MAX_BATCH_SIZE:
        raise ValueError(f"Batch size must be a positive integer less than {MAX_BATCH_SIZE}.")
    return text_list, batch_size

def batch_generator(text_list, batch_size, process_batch_fn):
    """Generate batches and process them with the provided function."""
    for i, batch in enumerate(range(0, len(text_list), batch_size)):
        batch_texts = text_list[batch : batch + batch_size]
        yield json.dumps({
            "batch_id": i,
            "result": process_batch_fn(batch_texts)
        }) + "\n"
    yield json.dumps({"status": "DONE"}) + "\n"

# Use nltk tokenize sentences
def sentence_tokenize(original_fulltext_batch: List[str]) -> List[List[str]]:
    return [sent_tokenize(text) for text in original_fulltext_batch]

# Use SimCSE to rank sentences based on similarity to the fulltext
def rank_sentences(tokenized_sentence_batch: List[List[str]], original_fulltext_batch: List[str]) -> List[List[dict]]:
    result = []
    for tokenized_sentence, original_text in zip(tokenized_sentence_batch, original_fulltext_batch):
        sent_sim_scores_by_simcse = model.similarity(tokenized_sentence, [original_text])
        sent_ids_select_by_simcse = np.argsort(-sent_sim_scores_by_simcse[:, 0]).tolist()
        result.append([
            {"id": i, "sentence": tokenized_sentence[i], "similarity_score": float(sent_sim_scores_by_simcse[i, 0])}
            for i in sent_ids_select_by_simcse
        ])
    return result

'''
Routers for APIs
'''
@simcse_bp.route("/nltk_sentence_tokenize", methods=["POST"])
@handle_api_errors
def api_sentence_tokenize():
    data = request.get_json()
    text_list, batch_size = validate_request(data)

    def process_batch_fn(batch_texts):
        return sentence_tokenize(batch_texts)
    
    return Response(
        stream_with_context(batch_generator(text_list, batch_size, process_batch_fn)),
        mimetype='application/jsonlines'
    )
    
    
@simcse_bp.route("/rank_by_fulltext", methods=["POST"])
@handle_api_errors
def api_rank_sentences():
    data = request.get_json()
    text_list, batch_size = validate_request(data)
    
    def process_batch_fn(batch_texts):
        tokenized_sentences = sentence_tokenize(batch_texts)
        return rank_sentences(tokenized_sentences, batch_texts)

    return Response(
        stream_with_context(batch_generator(text_list, batch_size, process_batch_fn)),
        mimetype='application/jsonlines'
    )
