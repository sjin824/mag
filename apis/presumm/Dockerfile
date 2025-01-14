FROM meg-base:1.0

# 设置工作目录
WORKDIR /app

# 拷贝依赖文件和代码
# COPY shared/shared_requirements.txt ./shared_requirements.txt
COPY requirements.txt ./requirements.txt

# 安装依赖，包括从 GitHub 安装 SimCSE
# RUN pip install --no-cache-dir -r shared_requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝代码
COPY . .
