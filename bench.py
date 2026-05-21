import os
from time import perf_counter
from random import randint, seed

import numpy as np

from nanovllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024
    # Start with 512 tokens (2 blocks) as a balanced chunk size.
    prefill_chunk_size = 512
    prefill_decode_mix = False
    decode_max_num_seqs = 0
    enable_profiling = True
    profiling_interval_s = 1.0
    profiling_path = "./profiling.jsonl"

    path = os.path.expanduser("/home/bstrike/huggingface/models--Qwen--Qwen3-1.7B")
    llm = LLM(
        path,
        enforce_eager=False,
        max_model_len=4096,
        gpu_memory_utilization=0.80,
        prefill_chunk_size=prefill_chunk_size,
        prefill_decode_mix=prefill_decode_mix,
        decode_max_num_seqs=decode_max_num_seqs,
        enable_profiling=enable_profiling,
        profiling_interval_s=profiling_interval_s,
        profiling_path=profiling_path,
    )

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm 解除下面注释使用原版vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    seq_ids = [llm.add_request(p, sp) for p, sp in zip(prompt_token_ids, sampling_params)]

    prefill_time = 0.0
    decode_time = 0.0
    decode_tokens = 0
    t0 = perf_counter()
    while not llm.is_finished():
        _, prefill_tokens, step_prefill_time, step_decode_tokens, step_decode_time = llm.step()
        # Use per-run timings from the engine to separate prefill/decode costs.
        if prefill_tokens > 0:
            prefill_time += step_prefill_time
        if step_decode_tokens > 0:
            decode_time += step_decode_time
            decode_tokens += step_decode_tokens
    total_time = perf_counter() - t0

    seqs = [llm.seqs[seq_id] for seq_id in seq_ids]
    ttft = [seq.first_token_time - seq.submit_time for seq in seqs if seq.first_token_time is not None]
    ttft_p50 = float(np.percentile(ttft, 50)) if ttft else 0.0
    ttft_p99 = float(np.percentile(ttft, 99)) if ttft else 0.0
    total_tokens = sum(seq.num_completion_tokens for seq in seqs)
    throughput = total_tokens / total_time if total_time > 0 else 0.0
    tpot_mean = (decode_time / decode_tokens) if decode_tokens > 0 else 0.0

    print("Metrics")
    print(f"concurrency: {num_seqs}")
    print(f"ttft_p50: {ttft_p50 * 1000:.2f} ms")
    print(f"ttft_p99: {ttft_p99 * 1000:.2f} ms")
    print(f"tpot_mean: {tpot_mean * 1000:.4f} ms")
    print(f"throughput: {throughput:.2f} tok/s")


if __name__ == "__main__":
    main()
