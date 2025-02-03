# api.py

from flask import Blueprint, request, jsonify

# 1) 创建一个蓝图对象，带有 url_prefix="/llm"
bp_llm = Blueprint("llm", __name__, url_prefix="/llm")

# 2) 定义一个字典，用来注册/存储 LLM 服务或生成器等对象
llm_services = {}
# 例如：llm_services["generator"] = <Generator 对象>

@bp_llm.route("/infer", methods=["POST"])
def infer():
    """
    POST /llm/infer
    请求体格式：
    {
      "dialogs": [
        [ {"role":"user", "content":"Hello..."} ],
        [ {"role":"user", "content":"..."} ],
        ...
      ]
    }
    """
    data = request.get_json()
    dialogs = data.get("dialogs", [])

    # 从 llm_services 字典中取出模型生成器
    generator = llm_services.get("generator", None)
    if generator is None:
        return jsonify({"error": "Model is not loaded"}), 500

    # 调用分布式推理
    results = generator.chat_completion(
        dialogs=dialogs,
        max_gen_len=512,
        temperature=0.6,
        top_p=0.9
    )
    return jsonify(results)
