from flask import Flask, request, jsonify
import importlib
import os
import requests

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return {"message": "Gateway is running!"}, 200

# # 动态加载所有 API
# def register_apis(app):
#     api_dir = "apis"
#     for api_name in os.listdir(api_dir):
#         api_path = os.path.join(api_dir, api_name)
#         if os.path.isdir(api_path):
#             try:
#                 # 动态加载蓝图
#                 module = importlib.import_module(f"apis.{api_name}.api")
#                 blueprint = getattr(module, f"{api_name}_bp")
#                 app.register_blueprint(blueprint, url_prefix=f"/{api_name}")
#                 print(f"Registered {api_name} at /{api_name}")
#             except Exception as e:
#                 print(f"Error registering {api_name}: {e}")

# register_apis(app)

@app.route("/simcse/rank_sentences", methods=["POST"])
def simcse_rank_sentences():
    try:
        # 转发请求到 simcse_api
        response = requests.post("http://simcse_api:5000/rank_sentences", json=request.get_json())
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return {"error": str(e)}, 500
    
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
