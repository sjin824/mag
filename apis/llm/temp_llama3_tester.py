import requests

dialogs = [
    # 对话1：只有一条用户消息
    [{"role": "user", "content": "Hello, can you tell me a joke?"}],
    # 对话2：用户提问
    [{"role": "user", "content": "What's the capital of France?"}],
    
    # 对话3：包含系统指令和用户消息，要求回答采用 Haiku 格式
    [
        {"role": "system", "content": "Answer in haiku."},
        {"role": "user", "content": "Describe a sunset."}
    ],
    [{"role": "user", "content": "You are a Large Language Model, Do you know who I am?"}],
]

resp = requests.post(
    "http://127.0.0.1:5000/llm/infer",
    json={"dialogs": dialogs}
)
print(resp.json())
