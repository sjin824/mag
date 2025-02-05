# import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
# from typing import List, Optional
# import os
# import sys


# def _setup_llama_import(llama_path):
#     if llama_path not in sys.path:
#         sys.path.append(llama_path)
    
# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '29500'  # 可以根据需要修改
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)


# def cleanup():
#     dist.destroy_process_group()


# def worker_fn(rank, world_size, llama_path, ckpt_dir, tokenizer_path, temperature, top_p, max_seq_len, max_batch_size, max_gen_len):
#     # 设置环境
#     setup(rank, world_size)
#     _setup_llama_import(llama_path)
#     from llama import Dialog, Llama
#     # 模型构建
#     device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
#     generator = Llama.build(
#         ckpt_dir=ckpt_dir,
#         tokenizer_path=tokenizer_path,
#         max_seq_len=max_seq_len,
#         max_batch_size=max_batch_size,
#     )
#     # 使用 DDP 包装模型
#     generator = DDP(generator, device_ids=[rank])

#     # 示例对话
#     dialogs: List[Dialog] = [
#         [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
#         [
#             {"role": "user", "content": "I am going to Paris, what should I see?"},
#             {
#                 "role": "assistant",
#                 "content": """\
# Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

# 1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
# 2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
# 3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

# These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
#             },
#             {"role": "user", "content": "What is so great about #1?"},
#         ],
#         [
#             {"role": "system", "content": "Always answer with Haiku"},
#             {"role": "user", "content": "I am going to Paris, what should I see?"},
#         ],
#         [
#             {
#                 "role": "system",
#                 "content": "Always answer with emojis",
#             },
#             {"role": "user", "content": "How to go from Beijing to NY?"},
#         ],
#     ]

#     # 生成结果
#     results = generator.module.chat_completion(
#         dialogs,
#         max_gen_len=max_gen_len,
#         temperature=temperature,
#         top_p=top_p,
#     )

#     # 打印结果
#     if rank == 0:  # 只在主进程打印结果
#         for dialog, result in zip(dialogs, results):
#             for msg in dialog:
#                 print(f"{msg['role'].capitalize()}: {msg['content']}\n")
#             print(
#                 f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
#             )
#             print("\n==================================\n")

#     cleanup()  # 清理进程组


# def main(
#         world_size,
#         llama_path,
#         ckpt_dir,
#         tokenizer_path,
#         temperature,
#         top_p,
#         max_seq_len,
#         max_batch_size,
#         max_gen_len
# ):
#     _setup_llama_import(llama_path)  # 本地导入 llama 模块
#     # 使用多进程管理
#     mp.spawn(
#         worker_fn,
#         args=(
#             world_size,
#             llama_path,
#             ckpt_dir,
#             tokenizer_path,
#             temperature,
#             top_p,
#             max_seq_len,
#             max_batch_size,
#             max_gen_len,
#         ),
#         nprocs=world_size,
#         join=True,
#     )


# if __name__ == "__main__":
#     main(
#         world_size = 1,
#         llama_path = '/data/shared/llama3/llama3',
#         ckpt_dir = '/data/shared/llama3/llama3/Meta-Llama-3-8B-Instruct',
#         tokenizer_path = '/data/shared/llama3/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model',
#         temperature = 0.6,
#         top_p = 0.9,
#         max_seq_len = 512,
#         max_batch_size = 4,
#         max_gen_len = 512
#     )

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import List, Optional
import os
import sys
import time

def _setup_llama_import(llama_path):
    if llama_path not in sys.path:
        sys.path.append(llama_path)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def worker_fn(rank, world_size, llama_path, ckpt_dir, tokenizer_path, temperature, top_p, max_seq_len, max_batch_size, max_gen_len):
    # 动态导入 llama 模块
    time1 = time.time()
    _setup_llama_import(llama_path)
    from llama import Dialog, Llama  # 子进程中动态导入
    
    time2 = time.time()
    # 设置分布式环境
    setup(rank, world_size)
    
    time3 = time.time()
    # 模型构建
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # 示例对话
    dialogs: List[Dialog] = [
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
        [
            {"role": "user", "content": "I am going to Paris, what should I see?"},
            {"role": "assistant", "content": "Eiffel Tower and the Louvre."},
            {"role": "user", "content": "What is so great about #1?"},
        ],
        [
            {"role": "system", "content": "Always answer with Haiku"},
            {"role": "user", "content": "I am going to Paris, what should I see?"},
        ],
    ]
    time4 = time.time()
    # 生成结果
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    # 只在 rank 0 打印结果
    if rank == 0:
        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
            print("\n==================================\n")
    print(f"rank={rank} time1={time2-time1} time2={time3-time2} time3={time4-time3} time4={time.time()-time4}")
    cleanup()  # 清理进程组

def main(
        world_size,
        llama_path,
        ckpt_dir,
        tokenizer_path,
        temperature,
        top_p,
        max_seq_len,
        max_batch_size,
        max_gen_len
):
    # 使用多进程管理
    mp.spawn(
        worker_fn,
        args=(
            world_size,
            llama_path,
            ckpt_dir,
            tokenizer_path,
            temperature,
            top_p,
            max_seq_len,
            max_batch_size,
            max_gen_len,
        ),
        nprocs=world_size,
        join=True,
    )

if __name__ == "__main__":
    main(
        world_size=1,
        llama_path='/data/shared/llama3/llama3',
        ckpt_dir='/data/shared/llama3/llama3/Meta-Llama-3-8B-Instruct',
        tokenizer_path='/data/shared/llama3/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model',
        temperature=0.6,
        top_p=0.9,
        max_seq_len=512,
        max_batch_size=4,
        max_gen_len=512
    )
