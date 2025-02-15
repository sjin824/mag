from typing import List, Union, Optional, Generator
import json
import requests
import argparse
import os

# Util for read_docs: Input formatting
def doc_2_list(document: Union[str, List[str]]) -> List[str]:
    '''
    Given a complete document in str/List[str] format, transfer into formatted json as standard input
    input: str or List[str]
        1. For a single str which is a coherent document, "aaa bbb ccc" -> ["aaa bbb ccc"]
        2. For a list of str as a coherent document, ["aaa", "bbb", "ccc"] -> ["aaa bbb ccc"]
    output: List[str]
    '''
    if isinstance(document, str):
        return [document]
    elif isinstance(document, list):
        return [" ".join(document)]

# Util for read_docs: Given a list of documents / or a document in .json format
def list_2_json(documents: List[str], batch: int = 1) -> str:
    '''
    ["aaa bbb ccc"] -> {... "text_list": ["aaa bbb ccc",], ...} in json format
    '''
    result = {
        "content": documents,
        "batch": batch
    }
    return json.dumps(result, indent=2)

# A Generator as main input: Given a list of documents / or a document in .json format
def read_docs(doc: Optional[str], doc_dir: Optional[str]) -> Generator[str, None, None]:
    if doc:
        yield json.loads(list_2_json(doc_2_list(doc)))
        
    if doc_dir:
        for _fn in os.listdir(doc_dir):
            if _fn.endswith(".json"):
                _fp = os.path.join(doc_dir, _fn)
                with open(_fp, 'r', encoding='utf-8') as file:
                    yield json.load(file)
                    
# Util for main: Get response from APIs
def get_response(api_url: str, json_data: dict):
    response = requests.post(
        api_url, 
        headers={"Content-Type": "application/json"},
        json=json_data
    )
    return response

# Main function
def main(
    url: str,
    documents: Generator[str, None, None]
):
    for document in documents:
        try:
            response = get_response(url, document)
            print(f'Processed: {response.text}')
            
        except Exception as e:
            print(f"Error document: {e}")
            continue
        
# Entry
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test APIs with documents.")
    parser.add_argument("--url", type=str, help="The URL of the tokenizer API. Default: %(default)s")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--doc", type=str, nargs='*', help="A string to test.")
    group.add_argument("--doc_dir", type=str, help="A directory or list of directories containing JSON documents.")

    args = parser.parse_args()
    
    documents = read_docs(args.doc, args.doc_dir)
    main(
        url=args.url,
        documents=documents
    )
    
# Conduct a test:
# python3 workflows/test_decont.py \
# --url "http://localhost:5002/nlp/xx" \
# --doc_dir "/home/sjin824/pyprojects/mag/tests/mock_data_for_decont"
