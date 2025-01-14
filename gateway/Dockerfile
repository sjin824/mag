# 使用共享基础镜像
FROM meg-base:1.0

# 设置工作目录
WORKDIR /app

# 拷贝网关的依赖文件
COPY requirements.txt .

# 安装网关的特定依赖
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝网关代码到容器
COPY . .

# 暴露网关的服务端口
EXPOSE 8000

# 启动网关服务
CMD ["python", "gateway.py"]
