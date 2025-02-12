from flask import Blueprint, request, jsonify, stream_with_context, Response
from functools import wraps
import json
from utils import validate_request, batch_generator

from handlers.spacy_handler import spacy_get_exts
from handlers.stanza_handler import stanza_get_exts
from handlers.simcse_handler import sentence_tokenize, rank_sentences
from handlers.decontextualizer_handler import decontextualize

nlp_bp = Blueprint('nlp', __name__, url_prefix="/nlp")

# Register available processors
NLP_PROCESSORS = {
    "spacy": spacy_get_exts,
    "stanza": stanza_get_exts,
    "simcse_tokenize": sentence_tokenize,
    "simcse_rank": lambda batch_texts: rank_sentences(sentence_tokenize(batch_texts), batch_texts),
    "decontextualize": decontextualize,
}

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

@nlp_bp.route("/<tool>", methods=["POST"])
@handle_api_errors
def api_handler(tool):
    """Unified API endpoint for all NLP processing tools."""
    if tool not in NLP_PROCESSORS:
        return jsonify({"error": f"Unknown tool: {tool}"}), 400

    data = request.get_json()
    content, batch_size = validate_request(data)
    process_fn = NLP_PROCESSORS[tool]

    return Response(
        stream_with_context(batch_generator(content, batch_size, process_fn)),
        mimetype='application/jsonlines'
    )
