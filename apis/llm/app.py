# app.py

import torch.distributed as dist
from flask import Flask
from api import bp_llm, llm_services
from handlers.llama3_handler import GeneratorController

def init_distributed():
    """
    使用 torch.distributed.init_process_group("nccl") 
    让多卡可以协同推理。
    该函数只会在 torchrun 等方式下正确获取 RANK / WORLD_SIZE。
    """
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    return rank

def create_app():
    """
    创建并返回一个 Flask app 对象，并注册蓝图。
    """
    app = Flask(__name__)
    app.register_blueprint(bp_llm)  # 把 /llm 相关路由挂到这个app
    return app

def main():
    # 1) 初始化分布式
    rank = init_distributed()

    # 2) 加载模型
    gen_controller = GeneratorController()
    gen_controller.load_model_cpkt()
    generator = gen_controller.get_generator()

    # 3) 将生成器对象放进 llm_services 字典
    llm_services["generator"] = generator

    if rank == 0:
        # 只有 rank=0 启动 HTTP 服务并监听端口
        print("Rank=0: 启动 Flask 服务...")
        app = create_app()
        app.run(host="0.0.0.0", port=5000, debug=False)
    else:
        # 其余 rank 进程只做 Worker，不开 Flask
        print(f"Rank={rank}: 进程保持等待，参与多卡推理...")
        # 如果没有其他逻辑，要保持进程不退出，可以 while True: pass
        while True:
            pass

if __name__ == "__main__":
    main()
