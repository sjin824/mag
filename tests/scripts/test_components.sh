#!/bin/bash

url_prefix="http://localhost:5002/nlp/"
urls=(
    "simcse_ranker" \
    "spacy_ner" \
    "stanza_ner" \
    "decontextualize" \
    "mixqg_question_gen" \
    "qa2claim_cg"
)

test_dir="/home/sjin824/pyprojects/mag/tests/mock_data/"
test_dir_paths=(
    "test_simcse" \
    "test_spacy" \
    "test_stanza" \
    "test_decontextualizer" \
    "test_mixqg" \
    "test_qa2claim"
)

for i in "${!urls[@]}"; do  
    url="${urls[$i]}"
    path="${test_dir_paths[$i]}"
    echo "========== Testing ${url} =========="
    python3 tests/scripts/test_component.py \
        --url "${url_prefix}${url}" \
        --doc_dir "${test_dir}${path}"
done