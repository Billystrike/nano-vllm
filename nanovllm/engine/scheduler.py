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
        self._zero_token_promotions = 0
        self._self_preempts = 0
        self._victim_preempts = 0

    def reset_stats(self):
        self._zero_token_promotions = 0
        self._self_preempts = 0
        self._victim_preempts = 0

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule_prefill(self) -> list[Sequence]:
        # Prefer shorter remaining prompts to improve block utilization and completion speed.
        if len(self.waiting) > 1:
            self.waiting = deque(sorted(self.waiting, key=lambda s: s.num_tokens - s.num_cached_tokens))
        def preempt_waiting_one() -> bool:
            for _ in range(len(self.waiting)):
                seq = self.waiting.pop()
                if seq.block_table:
                    self.block_manager.deallocate(seq)
                    seq.status = SequenceStatus.WAITING
                    seq.is_prefill = True
                    self.waiting.appendleft(seq)
                    return True
                self.waiting.append(seq)
            return False

        scheduled_seqs = []
        num_batched_tokens = 0
        retries = 0

        while retries < 2:
            # prefill
            num_waiting = len(self.waiting)
            # NOTE: num_waiting prevents scheduling the same seq twice in one step.
            while self.waiting and len(scheduled_seqs) < self.max_num_seqs and num_waiting > 0:
                seq = self.waiting[0]
                remaining = self.max_num_batched_tokens - num_batched_tokens
                if remaining == 0:
                    break
                if not seq.block_table:#如果序列没有块表，说明是第一次调度，需要计算可以复用的块数量
                    cached_block_ids = self.block_manager.get_cached_block_ids(seq)
                    num_cached_blocks = len(cached_block_ids)
                    num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
                else:
                    cached_block_ids = []
                    num_tokens = seq.num_tokens - seq.num_cached_tokens
                # If all prompt tokens are already cached, move to running without a GPU step.
                if num_tokens == 0:
                    self._zero_token_promotions += 1
                    cached_tokens = (len(cached_block_ids) * self.block_size) if not seq.block_table else seq.num_cached_tokens
                    target_blocks = (cached_tokens + self.block_size - 1) // self.block_size
                    if not seq.block_table:
                        if not self.block_manager.can_allocate_prefill(cached_block_ids, target_blocks):
                            # Not enough free blocks to attach cached prefix; try another seq this step.
                            self.waiting.rotate(-1)
                            num_waiting -= 1
                            continue
                        self.block_manager.ensure_blocks_for_prefill(seq, cached_block_ids, target_blocks)
                    seq.status = SequenceStatus.RUNNING
                    self.waiting.popleft()
                    self.running.append(seq)
                    num_waiting -= 1
                    continue
                # Cap per-step work for this seq to enable chunked prefill.
                scheduled_tokens = min(num_tokens, remaining)
                if self.prefill_chunk_size > 0:
                    scheduled_tokens = min(scheduled_tokens, self.prefill_chunk_size)
                if scheduled_tokens == 0:
                    # No progress for this seq (usually block/remaining budget); try another.
                    self.waiting.rotate(-1)
                    num_waiting -= 1
                    continue
                # Decide how many KV blocks are needed to cover cached + scheduled tokens.
                cached_tokens = (len(cached_block_ids) * self.block_size) if not seq.block_table else seq.num_cached_tokens
                target_tokens = cached_tokens + scheduled_tokens
                target_blocks = (target_tokens + self.block_size - 1) // self.block_size
                if not seq.block_table:
                    # On-demand: only ensure blocks up to target_blocks, not full prompt length.
                    if not self.block_manager.can_allocate_prefill(cached_block_ids, target_blocks):
                        # This seq needs more blocks than available; try another seq.
                        self.waiting.rotate(-1)
                        num_waiting -= 1
                        continue
                    self.block_manager.ensure_blocks_for_prefill(seq, cached_block_ids, target_blocks)
                else:
                    if not self.block_manager.can_extend(seq, target_blocks):
                        # This seq needs more blocks than available; try another seq.
                        self.waiting.rotate(-1)
                        num_waiting -= 1
                        continue
                    self.block_manager.ensure_blocks_for_prefill(seq, cached_block_ids, target_blocks)
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
            if scheduled_seqs or not self.waiting:
                break
            if not preempt_waiting_one():
                break
            retries += 1
        return scheduled_seqs

    def _pop_preemptable(self, exclude_seq_ids: set[int]) -> Sequence | None:
        if not exclude_seq_ids:
            return self.running.pop() if self.running else None
        for _ in range(len(self.running)):
            candidate = self.running.pop()
            if candidate.seq_id in exclude_seq_ids:
                # Keep excluded seqs in the queue for later steps.
                self.running.appendleft(candidate)
            else:
                return candidate
        return None

    def schedule_decode(self, max_num_seqs: int | None = None, exclude_seq_ids: set[int] | None = None) -> list[Sequence]:
        scheduled_seqs = []
        exclude_seq_ids = exclude_seq_ids or set()
        limit = self.max_num_seqs if not max_num_seqs else min(max_num_seqs, self.max_num_seqs)
        num_running = len(self.running)
        while self.running and len(scheduled_seqs) < limit and num_running > 0:
            seq = self.running.popleft()
            if seq.seq_id in exclude_seq_ids:
                # Do not decode sequences that are in the current prefill batch.
                self.running.append(seq)
                num_running -= 1
                continue
            # Preempt if we cannot append KV for this seq.
            while not self.block_manager.can_append(seq):
                victim = self._pop_preemptable(exclude_seq_ids)
                if victim is None:
                    # No preemptable seqs; preempt self to free blocks.
                    self.preempt(seq)
                    self._self_preempts += 1
                    seq = None
                    break
                self.preempt(victim)
                self._victim_preempts += 1
            if seq is None:
                break
            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
            num_running -= 1
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
