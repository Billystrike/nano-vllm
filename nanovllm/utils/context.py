from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    is_mixed: bool = False
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    cu_seqlens_q: torch.Tensor | None = None#cumulative sequence lengths of the sequences in the batch used to index into q
    cu_seqlens_k: torch.Tensor | None = None#The cumulative sequence lengths of of the sequences in the batch, used to index into kv.
    max_seqlen_q: int = 0#Maximum query sequence length in the batch.
    max_seqlen_k: int = 0#Maximum key/value sequence length in the batch.
    slot_mapping: torch.Tensor | None = None#记录每个token在KV缓存中的位置索引。
    context_lens: torch.Tensor | None = None#记录每条序列的上下文长度。
    block_tables: torch.Tensor | None = None#记录每条序列使用KV block 索引表。
    prefill_block_tables: torch.Tensor | None = None
    decode_context_lens: torch.Tensor | None = None
    decode_block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    is_mixed=False,
    num_prefill_tokens=0,
    num_decode_tokens=0,
    prefill_block_tables=None,
    decode_context_lens=None,
    decode_block_tables=None,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        is_mixed,
        num_prefill_tokens,
        num_decode_tokens,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
        prefill_block_tables,
        decode_context_lens,
        decode_block_tables,
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
