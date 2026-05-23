import gc
import json
import os
from time import perf_counter, sleep
from random import randint, seed

import numpy as np
import torch

from nanovllm import LLM, SamplingParams


def run_bench(
    total_requests: int,
    arrival_interval_s: float,
    prefill_chunk_size: int,
    gpu_memory_utilization: float,
    max_num_batched_tokens: int,
    max_input_len: int,
    max_output_len: int,
    enable_profiling: bool,
    profiling_interval_s: float,
    profiling_path: str,
    seed_base: int,
):
    # Keep prompts deterministic per concurrency so configs compare fairly.
    seed(0)
    path = os.path.expanduser("/home/bstrike/huggingface/models--Qwen--Qwen3-1.7B")
    llm = None
    try:
        llm = LLM(
            path,
            enforce_eager=False,
            max_model_len=4096,
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            prefill_chunk_size=prefill_chunk_size,
            enable_profiling=enable_profiling,
            profiling_interval_s=profiling_interval_s,
            profiling_path=profiling_path,
        )

        prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(total_requests)]
        sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len)) for _ in range(total_requests)]

        llm.generate(["Benchmark: "], SamplingParams())
        seq_ids = []
        arrival_times = [i * arrival_interval_s for i in range(total_requests)]
        next_idx = 0

        total_step_time = 0.0
        decode_only_time = 0.0
        decode_only_tokens = 0
        t0 = perf_counter()
        while next_idx < total_requests or not llm.is_finished():
            now = perf_counter()
            elapsed = now - t0
            # Enqueue all requests whose arrival time has passed.
            while next_idx < total_requests and elapsed >= arrival_times[next_idx]:
                seq_ids.append(llm.add_request(prompt_token_ids[next_idx], sampling_params[next_idx]))
                next_idx += 1
            if not llm.is_finished():
                _, step_time, step_prefill_tokens, step_decode_tokens, is_decode_only = llm.step()
                total_step_time += step_time
                if is_decode_only and step_decode_tokens > 0:
                    decode_only_time += step_time
                    decode_only_tokens += step_decode_tokens
            else:
                # No active requests; wait for the next arrival.
                if next_idx < total_requests:
                    sleep(min(0.05, max(0.0, arrival_times[next_idx] - elapsed)))
        total_time = perf_counter() - t0

        seqs = [llm.seqs[seq_id] for seq_id in seq_ids]
        ttft = [seq.first_token_time - seq.submit_time for seq in seqs if seq.first_token_time is not None]
        ttft_p50 = float(np.percentile(ttft, 50)) if ttft else 0.0
        ttft_p99 = float(np.percentile(ttft, 99)) if ttft else 0.0
        total_tokens = sum(seq.num_completion_tokens for seq in seqs)
        throughput = total_tokens / total_time if total_time > 0 else 0.0
        tpot_mean = (decode_only_time / decode_only_tokens) if decode_only_tokens > 0 else 0.0
        return {
            "ttft_p50": ttft_p50 * 1000.0,
            "ttft_p99": ttft_p99 * 1000.0,
            "tpot": tpot_mean * 1000.0,
            "throughput": throughput,
        }
    finally:
        if llm is not None:
            llm.exit()
            del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    max_input_len = 1024
    max_output_len = 1024
    enable_profiling = False
    profiling_interval_s = 1.0
    profiling_path = "./profiling.jsonl"
    seed_base = 0

    total_requests = 16
    arrival_interval_s = 0.1
    configs = [
        {"label": "no_chunk", "prefill_chunk_size": 0},
        {"label": "chunk_512", "prefill_chunk_size": 512},
    ]

    results = []
    for cfg in configs:
        print(f"Running {cfg['label']} @ requests={total_requests}, interval={arrival_interval_s:.3f}s")
        metrics = run_bench(
            total_requests=total_requests,
            arrival_interval_s=arrival_interval_s,
            prefill_chunk_size=cfg["prefill_chunk_size"],
            gpu_memory_utilization=0.85,
            max_num_batched_tokens=16384,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            enable_profiling=enable_profiling,
            profiling_interval_s=profiling_interval_s,
            profiling_path=profiling_path,
            seed_base=seed_base,
        )
        metrics["requests"] = total_requests
        metrics["arrival_ms"] = arrival_interval_s * 1000.0
        metrics["config"] = cfg["label"]
        results.append(metrics)

    print(f"{'Config':<12} {'Reqs':>5} {'Arr(ms)':>7} {'TTFT p50':>10} {'TTFT p99':>10} {'TPOT':>8} {'Thpt':>8}")
    for r in sorted(results, key=lambda r: r["config"]):
        print(
            f"{r['config']:<12} {r['requests']:>5} {r['arrival_ms']:>7.0f} {r['ttft_p50']:>10.1f} {r['ttft_p99']:>10.1f} "
            f"{r['tpot']:>8.1f} {r['throughput']:>8.1f}"
        )

    print("\nRaw JSON")
    print(json.dumps(results, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
