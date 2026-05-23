import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384#单次前向传播中，整个批次所有序列的 Token 总数的最大上限。
    max_num_seqs: int = 512#同时处理的请求数量（并发数）的最大上限。
    max_model_len: int = 4096#模型所能支持的单条序列的最大长度（prefill+decode)
    prefill_chunk_size: int = 0#prefill阶段单个序列每 step 的最大token数，0表示不限制
    enable_profiling: bool = False#是否启用CUDA event profiling
    profiling_interval_s: float = 1.0#profiling结果的输出间隔(秒)
    profiling_path: str = "./profiling.jsonl"#profiling输出路径
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.prefill_chunk_size >= 0
        assert self.profiling_interval_s > 0
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
