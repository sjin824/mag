import json
import torch
from typing import Dict
from handlers.base import BaseHandler

MAX_BATCH_SIZE = 100

def validate_request(data):
    """Validate input data for API endpoints."""
    if not data:
        raise ValueError("Request body must be JSON.")
    content = data.get("content", [])
    batch_size = data.get("batch_size", 1)
    if not isinstance(content, list) or not content:
        raise ValueError("The api received an empty input content.")
    if not isinstance(batch_size, int) or batch_size < 1 or batch_size > MAX_BATCH_SIZE:
        raise ValueError(f"Batch size must be an integer between 1 and {MAX_BATCH_SIZE}.")
    return content, batch_size

def batch_generator(content, batch_size, process_batch_fn):
    """Generate batches and process them with the provided function."""
    for i, batch in enumerate(range(0, len(content), batch_size)):
        batch_texts = content[batch : batch + batch_size]
        yield json.dumps({
            "batch_id": i,
            "response": process_batch_fn(batch_texts)
        }) + "\n"
        
def greedy_load_handlers(handlers_dict: Dict[str, BaseHandler]):
    '''
        Given a config dict of handlers, 
        1. greedy set their services on all cuda;
        2. load all services of them;
        3. yield the loading status.
    '''
    device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for tool_name, handler in handlers_dict.items():
        handler.set_device(device_type)
        handler.load_service()
        yield json.dumps({"message": f"{tool_name} loaded on device {str(handler.device)}"}) + "\n"