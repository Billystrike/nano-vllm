import pickle
from time import perf_counter
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)#在分布式环境中，每个进程通常负责一个GPU，因此需要设置当前进程使用的GPU设备。这里根据进程的rank来设置对应的GPU设备，确保每个进程在正确的GPU上运行。
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()#预热推理，获取模型推理的显存峰值等数据
        self.allocate_kv_cache()#根据显存信息来确定kvcache_blocks的数量并分配好模型的KV cache，使用Paged Attention的分块思想
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()#清空GPU缓存，释放未使用的内存，以便更准确地测量模型占用的内存。
        torch.cuda.reset_peak_memory_stats()#重置GPU的峰值内存统计数据，以便在后续测量中获得准确的峰值内存使用情况。
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]#创建长度为seq_len的全0 token序列，数量为num_seqs，作为模型的输入进行预热。预热的目的是让模型加载到GPU上，并进行一次前向传播，以便后续的内存分配和性能测量更准确。
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len#设置每条序列要计算的token数量为seq_len，等于输入序列的长度。
        self.run(seqs, True)
        torch.cuda.empty_cache()#再次清空GPU缓存，释放预热过程中占用的内存，以便正式运行时有更多可用内存。

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize#debug时显示占用显存量28MB
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():#将模型每一层都换成使用分块KV cache的版本，并把分配好的KV cache切片传给每一层
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens#已缓存的token数量
            seqlen_q = seq.num_scheduled_tokens#本次要计算的token数量
            end = start + seqlen_q#本次prefill的结束位置
            seqlen_k = end#参与注意力的key/value的数量
            input_ids.extend(seq[start:end])#将每条序列要计算的token拼接到一起，形成一个大batch。这样做是为了配合 FlashAttention 的 varlen 接口，
            positions.extend(range(start, end))#保存每个token 的位置索引
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)#cu_seqlens_q[-1]是前面所有序列的token数量，cu_seqlens_q[-1] + seqlen_q就是加上当前序列要计算的token数量，得到当前序列结束位置的索引。cu_seqlens_q的最后一个元素就是总的token数量。
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)#同理，cu_seqlens_k的最后一个元素是所有序列参与注意力的key/value的总数量。
            max_seqlen_q = max(seqlen_q, max_seqlen_q)#记录本次prefill中单条序列要计算的最大token数量，作为后续cudagraph的输入维度之一。
            max_seqlen_k = max(seqlen_k, max_seqlen_k)#记录本次prefill中单条序列参与注意力的key/value的最大数量，作为后续cudagraph的输入维度之一。
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:#如果是起始块，需要加上块内偏移量，确保slot_mapping中每个token对应到正确的物理槽位
                    slot_start += start % self.block_size
                if i != end_block - 1:#如果不是结束块，说明这个块内的token都要计算；
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:#如果是结束块，说明这个块内只有前面一部分token要计算，slot_end需要加上块内偏移量。
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)#如果是prefill阶段，准备输入ID和位置索引；如果是decode阶段，准备输入ID（上一个token）和位置索引（当前序列长度-1）
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None#如果是rank 0，准备温度张量；否则为None，因为只有rank 0需要进行采样。
        logits = self.run_model(input_ids, positions, is_prefill)#运行模型得到logits，如果是prefill阶段，logits的形状为[总token数量, vocab_size]；如果是decode阶段，logits的形状为[batch_size, vocab_size]。
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None#如果是rank 0，根据logits和温度张量进行采样，得到token ID；否则为None。
        reset_context()
        return token_ids

    def run_profiled(self, seqs: list[Sequence], is_prefill: bool) -> tuple[list[int] | None, dict]:
        # Profile CPU prepare time separately from GPU compute.
        t_prepare = perf_counter()
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        prepare_ms = (perf_counter() - t_prepare) * 1000.0

        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        # Use CUDA events to measure GPU time for forward and sampling.
        use_cuda_events = torch.cuda.is_available()
        if use_cuda_events:
            run_start = torch.cuda.Event(enable_timing=True)
            run_end = torch.cuda.Event(enable_timing=True)
            run_start.record()
            logits = self.run_model(input_ids, positions, is_prefill)
            run_end.record()
        else:
            t_run = perf_counter()
            logits = self.run_model(input_ids, positions, is_prefill)
            run_ms = (perf_counter() - t_run) * 1000.0

        sampler_ms = 0.0
        if self.rank == 0:
            if use_cuda_events:
                sample_start = torch.cuda.Event(enable_timing=True)
                sample_end = torch.cuda.Event(enable_timing=True)
                sample_start.record()
                token_ids = self.sampler(logits, temperatures).tolist()
                sample_end.record()
            else:
                t_sample = perf_counter()
                token_ids = self.sampler(logits, temperatures).tolist()
                sampler_ms = (perf_counter() - t_sample) * 1000.0
        else:
            token_ids = None

        if use_cuda_events:
            # Ensure all kernels finish before reading elapsed times.
            torch.cuda.synchronize()
            run_ms = run_start.elapsed_time(run_end)
            if self.rank == 0:
                sampler_ms = sample_start.elapsed_time(sample_end)

        reset_context()
        profile = {
            "prepare_ms": prepare_ms,
            "run_model_ms": run_ms,
            "sampler_ms": sampler_ms,
        }
        return token_ids, profile

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size#根据模型的最大长度和块大小，计算出每条序列最多需要多少个KV块。
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)#告诉注意力层“本步需要写入KV Cache的每个token对应到哪个物理槽位”
        context_lens = torch.zeros(max_bs, dtype=torch.int32)#每条序列的上下文长度，在decode的时候计算注意力的有效长度
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)#paged KV Cache的块索引表
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))#[1, 2, 4, 8, 16, 32, 48...max_bs]
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):#逆序取，bs从最大的值开始可以让 graph_pool 一开始就足够大，后续小 bs 直接复用内存池，减少内存重新分配和碎片化风险。
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()
        #捕获图所依赖的固定输入/输出缓冲区集合,在 decode 运行时把当前 step 的 input_ids/positions/slot_mapping/context_lens/block_tables 写进 
        # graph_vars 对应的切片,再直接 graph.replay()。graph 捕获时张量地址必须固定，运行时只能改内容不能改对象，所以需要用 graph_vars 来持有这些固定张量
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
