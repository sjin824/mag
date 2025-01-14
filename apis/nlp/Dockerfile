FROM meg-base:1.0

# 设置工作目录
WORKDIR /app

# 拷贝依赖文件和代码
COPY requirements.txt ./requirements.txt

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# Load apacy & stanza models
COPY load_model.py ./
RUN python load_model.py

# 拷贝代码
COPY . .

CMD ["python", "app.py"]