#!/bin/bash

# 定义 API 的 URL
SIMCSE_TOKENIZE_API="http://localhost:5002/nlp/simcse_tokenize"
SIMCSE_RANK_API="http://localhost:5002/nlp/simcse_rank"
SPACY_API="http://localhost:5002/nlp/spacy"
STANZA_API="http://localhost:5002/nlp/stanza"


# 遍历 mock_data 文件夹中的所有 JSON 文件
for json_file in tests/mock_data/*.json; do
  echo -e "=====================================\n"
  echo "Testing with $json_file..."
  # 使用 curl 发送 POST 请求。如果只想看元信息不关心响应内容则加上-o /dev/null
  echo -e "-------------------------------------"
  echo "Testing simcse_tokenize API..."
  curl -s -w "%{http_code}\n" -H "Content-Type: application/json" -d @"$json_file" "$SIMCSE_TOKENIZE_API"
  echo -e "-------------------------------------"
  echo "Testing simcse_rank API..."
  curl -s -w "%{http_code}\n" -H "Content-Type: application/json" -d @"$json_file" "$SIMCSE_RANK_API"
  echo -e "-------------------------------------"
  echo "Testing spacy API..."
  curl -s -w "%{http_code}\n" -H "Content-Type: application/json" -d @"$json_file" "$SPACY_API"
  echo -e "-------------------------------------"
  echo "Testing stanza API..."
  curl -s -w "%{http_code}\n" -H "Content-Type: application/json" -d @"$json_file" "$STANZA_API"
done
