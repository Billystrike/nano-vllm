from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.prefill_chunk_size = config.prefill_chunk_size
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule_prefill(self) -> list[Sequence]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        num_waiting = len(self.waiting)
        # NOTE: num_waiting prevents scheduling the same seq twice in one step.
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs and num_waiting > 0:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:#如果序列没有块表，说明是第一次调度，需要计算可以复用的块数量
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # Cap per-step work for this seq to enable chunked prefill.
            scheduled_tokens = min(num_tokens, remaining)
            if self.prefill_chunk_size > 0:
                scheduled_tokens = min(scheduled_tokens, self.prefill_chunk_size)
            if scheduled_tokens == 0:
                break
            if not seq.block_table:#如果序列没有块表，说明是第一次调度，需要分配块
                # Allocate the full block table once (not on-demand per chunk).
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = scheduled_tokens
            num_batched_tokens += seq.num_scheduled_tokens
            self.waiting.popleft()
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
            else:
                self.waiting.append(seq)
            scheduled_seqs.append(seq)
            num_waiting -= 1
        return scheduled_seqs

    def schedule_decode(self, max_num_seqs: int | None = None) -> list[Sequence]:
        scheduled_seqs = []
        limit = self.max_num_seqs if not max_num_seqs else min(max_num_seqs, self.max_num_seqs)
        while self.running and len(scheduled_seqs) < limit:
            seq = self.running.popleft()
            # Preempt if we cannot append KV for this seq.
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        if scheduled_seqs:
            self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = self.schedule_prefill()
        if scheduled_seqs:
            return scheduled_seqs, True
        scheduled_seqs = self.schedule_decode()
        assert scheduled_seqs
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):#当解码阶段的序列无法继续调度时，调用该函数将其抢占回预填充阶段，以腾出块资源给其他序列
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool, now: float | None = None):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            if now is not None and seq.first_token_time is None:
                seq.first_token_time = now
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                if now is not None:
                    seq.end_time = now
