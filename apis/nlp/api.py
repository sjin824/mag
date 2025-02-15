from flask import Blueprint, request, jsonify, stream_with_context, Response
from functools import wraps
import json
from utils import validate_request, batch_generator, greedy_load_handlers

from handlers.simcse_ranker import SimcseRankerHandler
from handlers.spacy_ner import SpacyNERHandler
from handlers.stanza_ner import StanzaNERHandler
from handlers.decontextualizer import DecontextualizerHandler
from handlers.mixqg_qg import MixQGHandler
from handlers.qa2claim_cg import QA2ClaimHandler
from handlers.docnli import DocNLIHandler

nlp_bp = Blueprint('nlp', __name__, url_prefix="/nlp")

# Register available handlers.
HANDLERS = {
    "simcse_ranker": SimcseRankerHandler({"hello": "world"}),
    "spacy_ner": SpacyNERHandler({"hello": "world"}),
    "stanza_ner": StanzaNERHandler({"hello": "world"}),
    "decontextualize": DecontextualizerHandler({'model_path': '/app/model_pts/t5_base/1611267950'}),
    "mixqg_question_gen": MixQGHandler({"hello": "world"}),
    "qa2claim_cg": QA2ClaimHandler({"hello": "world"}),
    "docnli": DocNLIHandler({'model_ckpt_path': "/app/model_pts/DocNLI.pretrained.RoBERTA.model.pt", "pretrain_model_dir": "roberta-large"})
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

@nlp_bp.route("/load_all_tools", methods=["POST"])
def load_all():
    return Response(
        stream_with_context(greedy_load_handlers(HANDLERS)),
        mimetype='application/jsonlines'
    )
            
@nlp_bp.route("/<tool>", methods=["POST"])
@handle_api_errors
def api_handler(tool):
    """Unified API endpoint for all NLP processing tools.
        1. Check if the tool is registered;
        2. validate the request;
        3. batch the request, run on the handler;
        4. stream the response.
    """
    if tool not in HANDLERS:
        return jsonify({"error": f"Unknown tool: {tool}"}), 400
    data = request.get_json()
    content, batch_size = validate_request(data)
    handler = HANDLERS[tool]
    process_fn = lambda batch: handler.process(batch)
    return Response(
        stream_with_context(batch_generator(content, batch_size, process_fn)),
        mimetype='application/jsonlines'
    )
