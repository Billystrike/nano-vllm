from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()#从空闲块列表中取出一个块ID
        block = self.blocks[block_id]#根据块ID获取块对象
        assert block.ref_count == 0#确保块未被使用
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:#如果块的哈希值不为-1且哈希值对应的块ID是当前块ID，说明该块之前被使用过，需要从哈希表中删除对应关系
            del self.hash_to_block_id[block.hash]
        block.reset()#重置块的状态，ref_count设为1，hash设为-1，token_ids清空
        self.used_block_ids.add(block_id)#将块ID添加到已使用块ID集合中
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def get_cached_block_ids(self, seq: Sequence) -> list[int]:
        # Find the longest prefix of full blocks that can be reused.
        h = -1
        cached_block_ids: list[int] = []
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            cached_block_ids.append(block_id)
        return cached_block_ids

    def can_allocate_prefill(self, cached_block_ids: list[int], target_blocks: int) -> bool:
        # Only blocks already in use do not consume free blocks.
        num_new_blocks = target_blocks
        for block_id in cached_block_ids[:target_blocks]:
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        return len(self.free_block_ids) >= num_new_blocks

    def can_extend(self, seq: Sequence, target_blocks: int) -> bool:
        # For an existing seq, we only need to allocate new blocks to reach target_blocks.
        needed = max(0, target_blocks - len(seq.block_table))
        return len(self.free_block_ids) >= needed

    def ensure_blocks_for_prefill(self, seq: Sequence, cached_block_ids: list[int], target_blocks: int):
        # First time: attach cached blocks and set cached token count.
        if not seq.block_table:
            for block_id in cached_block_ids:
                block = self.blocks[block_id]
                if block_id in self.used_block_ids:
                    block.ref_count += 1
                else:
                    block.ref_count = 1
                    self.free_block_ids.remove(block_id)
                    self.used_block_ids.add(block_id)
                seq.block_table.append(block_id)
            seq.num_cached_tokens = len(cached_block_ids) * self.block_size
        # Allocate only the blocks needed for the current prefill target.
        while len(seq.block_table) < target_blocks:
            seq.block_table.append(self._allocate_block())

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
