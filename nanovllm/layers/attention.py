import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape#N:写入kv_cache的token数
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1#检查key和value的最后一个维度是连续的，即head_dim维度是连续存储的
    assert key.stride(1) == head_dim and value.stride(1) == head_dim#检查key和value的第二个维度是head_dim的倍数，即num_heads维度是连续存储的
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)#将当前步的k和v存入当前层的k_cache和v_cache中，slot_mapping指示了每个序列对应的块ID和块内位置
        if context.is_mixed:
            # Mixed batch: split tokens into prefill and decode segments.
            prefill_tokens = context.num_prefill_tokens
            decode_tokens = context.num_decode_tokens
            outputs = []
            if prefill_tokens > 0:
                q_prefill = q[:prefill_tokens]
                k_prefill = k[:prefill_tokens]
                v_prefill = v[:prefill_tokens]
                if context.prefill_block_tables is not None:    # prefix cache
                    k_prefill, v_prefill = k_cache, v_cache
                o_prefill = flash_attn_varlen_func(
                    q_prefill,
                    k_prefill,
                    v_prefill,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=context.prefill_block_tables,
                )
                outputs.append(o_prefill)
            if decode_tokens > 0:
                q_decode = q[prefill_tokens:prefill_tokens + decode_tokens]
                o_decode = flash_attn_with_kvcache(
                    q_decode.unsqueeze(1),
                    k_cache,
                    v_cache,
                    cache_seqlens=context.decode_context_lens,
                    block_table=context.decode_block_tables,
                    softmax_scale=self.scale,
                    causal=True,
                )
                outputs.append(o_decode.squeeze(1))
            if len(outputs) == 1:
                return outputs[0]
            return torch.cat(outputs, dim=0)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
