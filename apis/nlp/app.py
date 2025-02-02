# apis/nlp/app.py
from flask import Flask
from api import nlp_bp  # 从 api.py 导入 blueprint

app = Flask(__name__)
url_prefix = "/nlp"

app.register_blueprint(nlp_bp, url_prefix=url_prefix)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
