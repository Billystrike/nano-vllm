import atexit
import json
import time
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        Sequence.block_size = config.kvcache_block_size
        self.seqs: dict[int, Sequence] = {}
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self._profiling_enabled = config.enable_profiling
        if self._profiling_enabled:
            # Accumulate profiling stats and flush to jsonl on an interval.
            self._profile_interval_s = config.profiling_interval_s
            self._profile_path = config.profiling_path
            self._profile_fh = open(self._profile_path, "a", encoding="utf-8")
            self._profile_last_flush = perf_counter()
            self._profile_acc = self._new_profile_acc()
        atexit.register(self.exit)

    def exit(self):
        if getattr(self, '_exited', False):
            return
        self._exited = True
        self.model_runner.call("exit")
        del self.model_runner
        if self._profiling_enabled:
            self._profile_fh.close()
        for p in self.ps:
            p.join()

    def _new_profile_bucket(self):
        return {
            "steps": 0,
            "tokens": 0,
            "prepare_ms": 0.0,
            "run_model_ms": 0.0,
            "sampler_ms": 0.0,
            "postprocess_ms": 0.0,
        }

    def _new_profile_acc(self):
        return {
            "prefill": self._new_profile_bucket(),
            "decode": self._new_profile_bucket(),
        }

    def _profile_add(self, kind: str, profile: dict, postprocess_ms: float, tokens: int):
        bucket = self._profile_acc[kind]
        bucket["steps"] += 1
        bucket["tokens"] += tokens
        bucket["prepare_ms"] += profile.get("prepare_ms", 0.0)
        bucket["run_model_ms"] += profile.get("run_model_ms", 0.0)
        bucket["sampler_ms"] += profile.get("sampler_ms", 0.0)
        bucket["postprocess_ms"] += postprocess_ms

    def _maybe_flush_profile(self):
        now = perf_counter()
        if now - self._profile_last_flush < self._profile_interval_s:
            return
        payload = {
            "ts": time.time(),
            "interval_s": now - self._profile_last_flush,
            "prefill": self._profile_acc["prefill"],
            "decode": self._profile_acc["decode"],
        }
        self._profile_fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self._profile_fh.flush()
        self._profile_last_flush = now
        self._profile_acc = self._new_profile_acc()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        seq.submit_time = perf_counter()
        self.seqs[seq.seq_id] = seq
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self):
        step_start = perf_counter()
        outputs = []
        prefill_tokens = 0
        decode_tokens = 0
        is_decode_only = False
        decode_limit = None
        prefill_seqs = self.scheduler.schedule_prefill()
        if prefill_seqs:
            # Prefill run (varlen) executes before decode in the same step.
            prefill_tokens = sum(seq.num_scheduled_tokens for seq in prefill_seqs)
            if self._profiling_enabled:
                token_ids, profile = self.model_runner.call("run_profiled", prefill_seqs, True)
            else:
                token_ids = self.model_runner.call("run", prefill_seqs, True)
            t_post = perf_counter()
            self.scheduler.postprocess(prefill_seqs, token_ids, True, perf_counter())#将计算得到的token_ids存入对应的序列中，并更新块表和状态
            postprocess_ms = (perf_counter() - t_post) * 1000.0
            if self._profiling_enabled:
                # Keep profiling separate from throughput timing to avoid changing behavior.
                self._profile_add("prefill", profile, postprocess_ms, prefill_tokens)
            outputs.extend([(seq.seq_id, seq.completion_token_ids) for seq in prefill_seqs if seq.is_finished])
        else:
            decode_seqs = self.scheduler.schedule_decode(decode_limit)
            if decode_seqs:
                # Decode run is a separate forward pass to keep prefill kernels unchanged.
                decode_tokens = len(decode_seqs)
                is_decode_only = True
                if self._profiling_enabled:
                    token_ids, profile = self.model_runner.call("run_profiled", decode_seqs, False)
                else:
                    token_ids = self.model_runner.call("run", decode_seqs, False)
                t_post = perf_counter()
                self.scheduler.postprocess(decode_seqs, token_ids, False, perf_counter())#将计算得到的token_ids存入对应的序列中，并更新块表和状态
                postprocess_ms = (perf_counter() - t_post) * 1000.0
                if self._profiling_enabled:
                    self._profile_add("decode", profile, postprocess_ms, decode_tokens)
                outputs.extend([(seq.seq_id, seq.completion_token_ids) for seq in decode_seqs if seq.is_finished])

        if self._profiling_enabled:
            self._maybe_flush_profile()
        step_time = perf_counter() - step_start
        return outputs, step_time, prefill_tokens, decode_tokens, is_decode_only

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            output, step_time, prefill_tokens, decode_tokens, _ = self.step()
            if prefill_tokens > 0 and step_time > 0:
                prefill_throughput = prefill_tokens / step_time
            if decode_tokens > 0 and step_time > 0:
                decode_throughput = decode_tokens / step_time
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
