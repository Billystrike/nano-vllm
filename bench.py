import gc
import json
from itertools import product
import os
from time import perf_counter
from random import randint, seed

import numpy as np
import torch

from nanovllm import LLM, SamplingParams


def run_bench(
    num_seqs: int,
    prefill_chunk_size: int,
    prefill_decode_mix: bool,
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
    seed(seed_base + num_seqs)
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
            prefill_decode_mix=prefill_decode_mix,
            decode_max_num_seqs=0,
            enable_profiling=enable_profiling,
            profiling_interval_s=profiling_interval_s,
            profiling_path=profiling_path,
        )

        prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
        sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len)) for _ in range(num_seqs)]

        llm.generate(["Benchmark: "], SamplingParams())
        seq_ids = [llm.add_request(p, sp) for p, sp in zip(prompt_token_ids, sampling_params)]

        total_step_time = 0.0
        decode_only_time = 0.0
        decode_only_tokens = 0
        llm.mixed_steps = 0
        llm.prefill_only_steps = 0
        llm.decode_only_steps = 0
        llm.scheduler.reset_stats()
        t0 = perf_counter()
        while not llm.is_finished():
            _, step_time, step_prefill_tokens, step_decode_tokens, is_decode_only = llm.step()
            total_step_time += step_time
            if is_decode_only and step_decode_tokens > 0:
                decode_only_time += step_time
                decode_only_tokens += step_decode_tokens
        total_time = perf_counter() - t0

        seqs = [llm.seqs[seq_id] for seq_id in seq_ids]
        ttft = [seq.first_token_time - seq.submit_time for seq in seqs if seq.first_token_time is not None]
        ttft_p50 = float(np.percentile(ttft, 50)) if ttft else 0.0
        ttft_p99 = float(np.percentile(ttft, 99)) if ttft else 0.0
        total_tokens = sum(seq.num_completion_tokens for seq in seqs)
        throughput = total_tokens / total_time if total_time > 0 else 0.0
        tpot_mean = (decode_only_time / decode_only_tokens) if decode_only_tokens > 0 else 0.0
        total_steps = llm.mixed_steps + llm.prefill_only_steps + llm.decode_only_steps

        return {
            "ttft_p50": ttft_p50 * 1000.0,
            "ttft_p99": ttft_p99 * 1000.0,
            "tpot": tpot_mean * 1000.0,
            "throughput": throughput,
            "total_steps": total_steps,
            "mixed_steps": llm.mixed_steps,
            "prefill_only": llm.prefill_only_steps,
            "decode_only": llm.decode_only_steps,
            "zero_token_promotions": llm.scheduler._zero_token_promotions,
            "self_preempts": llm.scheduler._self_preempts,
            "victim_preempts": llm.scheduler._victim_preempts,
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

    concurrencies = [8, 32, 64, 128, 256]
    configs = [
        {"label": "no_chunk", "prefill_chunk_size": 0, "prefill_decode_mix": False},
        {"label": "chunk_512", "prefill_chunk_size": 512, "prefill_decode_mix": True},
    ]

    results = []
    for concurrency, cfg in product(concurrencies, configs):
        print(f"Running {cfg['label']} @ concurrency={concurrency}")
        metrics = run_bench(
            num_seqs=concurrency,
            prefill_chunk_size=cfg["prefill_chunk_size"],
            prefill_decode_mix=cfg["prefill_decode_mix"],
            gpu_memory_utilization=0.85,
            max_num_batched_tokens=16384,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            enable_profiling=enable_profiling,
            profiling_interval_s=profiling_interval_s,
            profiling_path=profiling_path,
            seed_base=seed_base,
        )
        metrics["concurrency"] = concurrency
        metrics["config"] = cfg["label"]
        results.append(metrics)

    print(f"{'Config':<12} {'Conc':>5} {'TTFT p50':>10} {'TTFT p99':>10} {'TPOT':>8} {'Thpt':>8} {'Steps':>7}")
    for r in sorted(results, key=lambda r: (r["concurrency"], r["config"])):
        print(
            f"{r['config']:<12} {r['concurrency']:>5} {r['ttft_p50']:>10.1f} {r['ttft_p99']:>10.1f} "
            f"{r['tpot']:>8.1f} {r['throughput']:>8.1f} {r['total_steps']:>7}"
        )

    print("\nRaw JSON")
    print(json.dumps(results, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
