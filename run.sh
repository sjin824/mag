# 刷新组权限：
newgrp docker

# 构建基础镜像 
docker build -t project-base:latest -f shared/Dockerfile.base .
# 构建和运行服务，应自动构建所有服务的镜像并启动容器 # 因为有插件（docker compose version）所以不用docker-compose而是docker compose
docker compose up --build
# 如果成功启动，查看运行中的容器，且可以获取ID。-a 参数可查看所有容器，包括已停止的
docker ps

# 单独启动并测试simcse_api服务
docker build -t simcse-api ./apis/simcse_api
docker run -p 5001:5000 simcse-api

# 测试网关服务
curl -X POST http://localhost:8000/simcse/rank_sentences \
     -H "Content-Type: application/json" \
     -d '{"fulltext": "This is the first sentence. This is the second sentence."}'
# 如 SimCSE API服务被单独启动（见行8-10），使用 curl 直接测试对应端口
curl -X POST http://localhost:5001/rank_sentences \
     -H "Content-Type: application/json" \
     -d '{"fulltext": "This is the first sentence. This is the second sentence."}'

# 如果某个服务无法正常工作，使用以下命令查看其日志
# docker logs <container_id>

# 停止所有正在运行的容器，停止所有由当前 docker-compose.yml 文件管理的容器，并删除对应的网络。不会删除镜像和数据卷
docker compose down
# 删除某容器
# docker rm <container_id>
# 删除所有停止的容器
docker container prune

# 列出所有镜像，查看哪些是项目相关的
docker images
# 删除某镜像
# docker rmi <image_id>
# 删除所有未被使用的镜像（无容器关联的镜像）
docker image prune

# Docker 数据卷用于持久化数据，但如果你的项目没有使用数据卷，清理它们可以释放空间
# 列出所有数据卷
docker volume ls
# 删除未被使用的数据卷
docker volume prune

# 清理docker-compose的专属网络
# 列出所有网络
sudo docker network ls
# 删除未使用的网络
sudo docker network prune

# 一键清理未使用的资源
sudo docker system prune -a


############################################################################
# 1 构建
docker compose build --no-cache # --no-cache 可以确保每一步都重新拉取或构建，不使用任何层缓存，从而真正“干净”地构建镜像。

# 2 手动启动有GPU需求的部分，如simcse_api
# 需要确保docker-compose.yml文件中的simcse_api有指定镜像名image: simcse_api:1.0
# --name指定一个容器的自定义名称，不然会变成系统默认名
docker run -d --gpus all -p 5002:5000 --name simcse_gpu_run simcse_api:1.0
docker run -d --gpus all -p 5003:5000 --name presumm_gpu_run presumm_api:1.0
docker run -d --gpus all -p 5004:5000 --name nlp_gpu_run nlp_api:1.0

# 3 Compose启动 - 无GPU需求部分，如gateway。 -d代表后台运行
docker compose up -d gateway
