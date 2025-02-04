import os
import sys
from typing import List

# 默认的 llama3 路径以及可选模型缩写
LLAMA3PATH = '/data/shared/llama3/llama3'
MODELS = {
    '8S': "Meta-Llama-3-8B",
    '8I': "Meta-Llama-3-8B-Instruct",
    '70S': "Meta-Llama-3-70B",
    '70I': "Meta-Llama-3-70B-Instruct"
}

class GeneratorController:
    def __init__(self, llama3_path: str = LLAMA3PATH, model_abbr: str = '8I'):
        self.llama3_path = llama3_path
        self.model_abbr = model_abbr

        self.ckpt_path = os.path.join(self.llama3_path, MODELS[model_abbr])
        self.tokenizer_path = os.path.join(self.ckpt_path, 'tokenizer.model')
        self.generator = None

        self.setup_llama_import()

    def setup_llama_import(self):
        if self.llama3_path not in sys.path:
            sys.path.append(self.llama3_path)
        try:
            from llama import Dialog, Llama  # 本地导入 llama 模块
            self.Dialog = Dialog
            self.Llama = Llama
        except ImportError as e:
            raise ImportError(f"导入 llama 模块失败：{e}")

    def load_model_cpkt(self, max_seq_len: int = 8192, max_batch_size: int = 4):
        try:
            self.generator = self.Llama.build(
                ckpt_dir=self.ckpt_path,
                tokenizer_path=self.tokenizer_path,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
            )
            print(f"模型 {self.model_abbr} 从 {self.ckpt_path} 成功加载。")
        except Exception as e:
            raise RuntimeError(f"加载模型 checkpoint 失败：{e}")

    def get_generator(self):
        return self.generator
