from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))#表示每个偶数维度的旋转频率，频率随着维度增加而增加
        t = torch.arange(max_position_embeddings, dtype=torch.float)#表示一个token的绝对位置
        freqs = torch.einsum("i,j -> ij", t, inv_freq)#计算每个位置和每个频率的乘积，得到一个形状为 (max_position_embeddings, rotary_dim // 2) 的矩阵，其中每行表示一个绝对位置的旋转频率
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)#将 cos 和 sin 沿最后一个维度拼接，得到一个形状为 (max_position_embeddings, 1, rotary_dim) 的矩阵，其中每行表示一个绝对位置的旋转频率的余弦和正弦值。然后使用 unsqueeze_ 在第二个维度添加一个维度，以便在后续计算中与 query 和 key 的形状匹配。
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
