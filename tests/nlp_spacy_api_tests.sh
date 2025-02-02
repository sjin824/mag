#!/bin.bash

# 定义 API 的 URL
NLP_SPACY_API_URL="http://localhost:5004/nlp/spacy"
# NLP_STANZA_API_URL="http://localhost:5004/nlp/stanza"

# 遍历 mock_data 文件夹中的所有 JSON 文件
for json_file in tests/mock_data/*.json; do
  echo "Testing with $json_file..."
  # 使用 curl 发送 POST 请求。如果只想看元信息不关心响应内容则加上-o /dev/null
  echo -e "-------------------------------------"
  echo "Testing NLP spaCy API..."
  curl -s -w "%{http_code}\n" -H "Content-Type: application/json" -d @"$json_file" "$NLP_SPACY_API_URL"
  
  # echo "Testing NLP stanza API..."
  # curl -s -w "%{http_code}\n" -H "Content-Type: application/json" -d @"$json_file" "$NLP_STANZA_API_URL"
  echo -e "=====================================\n"
done
