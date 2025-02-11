import json

MAX_BATCH_SIZE = 100

def validate_request(data):
    """Validate input data for API endpoints."""
    if not data:
        raise ValueError("Request body must be JSON.")
    content = data.get("content", [])
    batch_size = data.get("batch", 10)
    if not isinstance(content, list) or not content:
        raise ValueError("Input must be a list of strings, and cannot be empty.")
    if not isinstance(batch_size, int) or batch_size <= 0 or batch_size > MAX_BATCH_SIZE:
        raise ValueError(f"Batch size must be a positive integer less than {MAX_BATCH_SIZE}.")
    return content, batch_size

def batch_generator(content, batch_size, process_batch_fn):
    """Generate batches and process them with the provided function."""
    for i, batch in enumerate(range(0, len(content), batch_size)):
        batch_texts = content[batch : batch + batch_size]
        yield json.dumps({
            "batch_id": i,
            "content": process_batch_fn(batch_texts)
        }) + "\n"
    # yield json.dumps({"status": "DONE"}) + "\n"
