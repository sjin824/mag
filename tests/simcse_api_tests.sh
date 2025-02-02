#!/bin.bash

# 定义 API 的 URL
NLTK_TOKENIZE_API_URL="http://localhost:5002/simcse/nltk_sentence_tokenize"
SIMCSE_RANK_API_URL="http://localhost:5002/simcse/rank_by_fulltext"

# 遍历 mock_data 文件夹中的所有 JSON 文件
for json_file in tests/mock_data/*.json; do
  echo "Testing with $json_file..."
  # 使用 curl 发送 POST 请求。如果只想看元信息不关心响应内容则加上-o /dev/null
  echo -e "-------------------------------------"
  echo "Testing NLTK Tokenize API..."
  curl -s -w "%{http_code}\n" -H "Content-Type: application/json" -d @"$json_file" "$NLTK_TOKENIZE_API_URL"
  echo -e "-------------------------------------"
  echo "Testing SimCSE Sentence Ranking API..."
  curl -s -w "%{http_code}\n" -H "Content-Type: application/json" -d @"$json_file" "$SIMCSE_RANK_API_URL"
  echo -e "-------------------------------------"
  echo -e "=====================================\n"
done
