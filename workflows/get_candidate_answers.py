import asyncio
from typing import List, Union, Optional, AsyncGenerator
import json
import aiohttp
import os

API_URLS = {
    "spacy": "http://localhost:5002/nlp/spacy",
    "stanza": "http://localhost:5002/nlp/stanza",
    "simcse_tokenize": "http://localhost:5002/nlp/simcse_tokenize",
    "simcse_rank": "http://localhost:5002/nlp/simcse_rank"
}

async def call_api_streaming(url: str, data: dict):
    decoder = json.JSONDecoder()
    buffer = ""
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as response:
            # if response.status != 200:
            #     raise Exception(f"Request failed with status {response.status}")
            async for chunk in response.content.iter_any():
                buffer += chunk.decode('utf-8')
                while True:
                    try:
                        result, index = decoder.raw_decode(buffer)
                        yield result
                        buffer = buffer[index:].lstrip()
                    except json.JSONDecodeError:
                        break

# Broadcast layer
async def broadcast_to_consumers(broadcast_q: asyncio.Queue, queue_list: List[asyncio.Queue]):
    while True:
        data = await broadcast_q.get()
        for queue in queue_list:
            await queue.put(data)
        broadcast_q.task_done()
        
# 1.1 Convert document to list of sentences
def doc_2_list(document: Union[str, List[str]]) -> List[str]:
    if isinstance(document, str):
        return [document]
    elif isinstance(document, list):
        return [" ".join(document)]

# 1.2 Convert list of documents to JSON string
def list_2_json(documents: List[str], batch: int = 1) -> str:
    result = {
        "content": documents,
        "batch": batch
    }
    return json.dumps(result, indent=2)

# 1. Read the documents
async def read_docs(doc: Optional[str], doc_dir: Optional[str]) -> AsyncGenerator[str, None]:
    if doc:
        yield json.loads(list_2_json(doc_2_list(doc)))
        
    if doc_dir:
        for _fn in os.listdir(doc_dir):
            if _fn.endswith(".json"):
                _fp = os.path.join(doc_dir, _fn)
                with open(_fp, 'r', encoding='utf-8') as file:
                    f_content = file.read()
                    yield json.loads(f_content)

# 2. Rank the sentences using SimCSE
async def simcse_ranking_top10(doc: Optional[str], doc_dir: Optional[str], output_q: asyncio.Queue):
    file_id = 0
    # print("=================== simcse start ===================")
    async for document in read_docs(doc, doc_dir):
        async for api_response in call_api_streaming(API_URLS["simcse_rank"], document):
            # 对每篇文章取前 10 个句子
            top_sentences = [[sentence_dict["sentence"] for sentence_dict in doc[:10]] for doc in api_response["content"]]

            result = {
                "file_id": file_id,
                **api_response,  # 自动解构 batch_id 和其他字段
                "top_sentences": top_sentences,  # 现在是按文章组织的列表
            }
            # print(f"simcse api top10 response: {result}\n")
            await output_q.put(result)
        file_id += 1

# 3.a Extract entities using spaCy
async def extract_entities_by_spacy(input_q: asyncio.Queue, output_q: asyncio.Queue):
    # print("=================== spacy start ===================")
    while True:
        _input = await input_q.get()
        joint_sentences_list = [" ".join(sentences) for sentences in _input["top_sentences"]]
        async for entities in call_api_streaming(API_URLS["spacy"], {"content": joint_sentences_list}):
            result = {**_input, "spacy_entities": entities["content"]}
            # print(f"spacy api response: {result}\n")
            await output_q.put(result)
        input_q.task_done()

# 3.b Extract entities using Stanza   
async def extract_entities_by_stanza(input_q: asyncio.Queue, output_q: asyncio.Queue):
    # print("=================== stanza start ===================")
    while True:
        _input = await input_q.get()
        joint_sentences_list = [" ".join(sentences) for sentences in _input["top_sentences"]]
        async for entities in call_api_streaming(API_URLS["stanza"], {"content": joint_sentences_list}):
            result = {**_input, "stanza_entities": entities["content"]}
            # print(f"stanza api response: {result}\n")
            await output_q.put(result)
        input_q.task_done()


# 3.2 Merge and deduplicate entities
async def merge_deduplicate(ner_q: asyncio.Queue, output_q: asyncio.Queue):
    temp = {}
    # print("=================== merge start ===================")
    while True:
        _input = await ner_q.get()
        key = (_input["file_id"], _input["batch_id"])
        if key not in temp:
            temp[key] = {}
        if "spacy_entities" in _input:
            temp[key]["spacy_entities"] = _input["spacy_entities"]
        if "stanza_entities" in _input:
            temp[key]["stanza_entities"] = _input["stanza_entities"]
            
        if "spacy_entities" in temp[key] and "stanza_entities" in temp[key]:
            spacy_entities = temp[key]["spacy_entities"]
            stanza_entities = temp[key]["stanza_entities"]
            
            assert len(spacy_entities) == len(stanza_entities), "Mismatch in article count between spaCy and Stanza."
            merged_entities = []
            for spacy_doc_entities, stanza_doc_entities in zip(spacy_entities, stanza_entities):
                merged_doc_entities = list(dict.fromkeys(spacy_doc_entities + stanza_doc_entities))
                merged_entities.append(merged_doc_entities)
                    
            result = {
                "file_id": _input["file_id"],
                "batch_id": _input["batch_id"],
                "top_sentences": _input["top_sentences"],
                "content": _input["content"],
                "entities": merged_entities
            }
            print(f"merge response: {result}\n")
            await output_q.put(merged_entities)
            del temp[key]
        ner_q.task_done()

async def main(doc: str, doc_dir: str):
    # Queue initialization
    broadcast_q = asyncio.Queue()
    spacy_q = asyncio.Queue()
    stanza_q = asyncio.Queue()
    broadcast_q_list = [spacy_q, stanza_q]
    ner_q = asyncio.Queue()
    merged_output_q = asyncio.Queue()

    
    # Tasks running
    producer_t = asyncio.create_task(simcse_ranking_top10(doc, doc_dir, broadcast_q))
    broadcast_t = asyncio.create_task(broadcast_to_consumers(broadcast_q, broadcast_q_list))
    spacy_t = asyncio.create_task(extract_entities_by_spacy(spacy_q, ner_q))
    stanza_t = asyncio.create_task(extract_entities_by_stanza(stanza_q, ner_q))
    merge_t = asyncio.create_task(merge_deduplicate(ner_q, merged_output_q))

    # Wait tasks
    await producer_t
    await broadcast_q.join()
    await spacy_q.join()
    await stanza_q.join()
    await ner_q.join()
    await merged_output_q.join()
    
    # 取消所有消费者任务（它们会自动退出）
    broadcast_t.cancel()
    spacy_t.cancel()
    stanza_t.cancel()
    merge_t.cancel()

    print("Pipeline completed.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test APIs with documents.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--doc", type=str, help="A document string to test.")
    group.add_argument("--doc_dir", type=str, help="A directory containing JSON documents.")

    args = parser.parse_args()
    asyncio.run(main(args.doc, args.doc_dir))
    
# Conduct a test:
# python3 workflows/main.py --doc_dir "/home/sjin824/pyprojects/mag/tests/mock_data/"
# python3 workflows/main.py --doc "Hello world. This is a test. This is the third sentence. Thank you."
# python3 workflows/get_candidate_answers.py --doc_dir "/home/sjin824/pyprojects/mag/tests/mock_data/"