FROM meg-base:1.0

# 设置工作目录
WORKDIR /app

# 拷贝+安装依赖文件 包括github上的simcse
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 下载nltk的punkt模型
RUN python -m nltk.downloader punkt punkt_tab

# 下载spaCy, stanza模型
RUN python -m spacy download en_core_web_lg

# Load spacy & stanza models
COPY load_model.py ./
RUN python load_model.py

# 拷贝代码
COPY . .

CMD ["python", "app.py"]