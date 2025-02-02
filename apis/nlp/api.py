from flask import Blueprint, request, jsonify, stream_with_context, Response
import spacy
import stanza
from functools import wraps 
import json

nlp_bp = Blueprint('nlp', __name__)
spacy_nlp = spacy.load('en_core_web_lg')
stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
MAX_BATCH_SIZE = 100

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
    
# Spacy: Extract Noun Chunks & Named Entities
def spacy_get_exts(batch_texts, is_noun_chunks=True, is_named_entities=True):
    results = []
    for spacy_doc in spacy_nlp.pipe(batch_texts):
        result = []
        if is_noun_chunks:
            result += [noun_chunk.text for noun_chunk in spacy_doc.noun_chunks]
        if is_named_entities:
            result += [named_entities.text for named_entities in spacy_doc.ents]
        results.append(result)
    return results

# Util for Stanza
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
    
# Stanza: Extract Noun & Verb Phrases, and NOUN, VERB, ADJ, ADV
def stanza_get_exts(batch_texts, is_NP=True, is_VP=True, is_upos=True):
    results = []
    for text in batch_texts:  # 对每个文本逐条处理
        stanza_doc = stanza_nlp(text)
        result = []
        for sent in stanza_doc.sentences:
            # 提取名词短语
            if is_NP:
                result.extend(get_phrases(sent.constituency, 'NP'))

            # 提取动词短语
            if is_VP:
                result.extend(get_phrases(sent.constituency, 'VP'))

            # 提取指定词性
            if is_upos:
                result.extend(
                    [word.text for word in sent.words if word.upos in ['NOUN', 'VERB', 'ADJ', 'ADV']]
                )

        results.append(result)  # 每个文本的提取结果是一个列表
    return results

'''
Routes
'''
@nlp_bp.route("/spacy", methods=["POST"])
@handle_api_errors
def api_spacy(): # 改名?
    data = request.get_json()
    text_list, batch_size = validate_request(data)

    def process_batch_fn(batch_texts):
        return spacy_get_exts(batch_texts, is_noun_chunks=True, is_named_entities=True)
    
    return Response(
        stream_with_context(batch_generator(text_list, batch_size, process_batch_fn)),
        mimetype='application/jsonlines'
    )
    
@nlp_bp.route("/stanza", methods=["POST"])
@handle_api_errors
def api_stanza(): # 改名?
    data = request.get_json()
    text_list, batch_size = validate_request(data)
    
    def process_batch_fn(batch_texts):
        return stanza_get_exts(batch_texts, is_NP=True, is_VP=True, is_upos=True)
    
    return Response(
        stream_with_context(batch_generator(text_list, batch_size, process_batch_fn)),
        mimetype='application/jsonlines'
    )