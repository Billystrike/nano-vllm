import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile#在程序首次运行时，将这段 Python 代码即时编译（JIT）成高效的底层机器码（通常使用 Triton 编写内核）
    #配合 @torch.compile，这种由基础算子（除法、指数、求最大值）组成的计算图可以被深度融合成一个高效的 GPU Kernel
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        '''
        logits形状通常为 [batch_size, vocab_size]，是模型最后一层输出的未归一化的原始分数。
        temperatures形状为 [batch_size]（或标量），控制生成的随机性。温度越高越随机，越低越确定。
        '''
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))#temperatures>1,除以T会使logits变小，生成更随机的结果；temperatures<1,除以T会使logits变大，生成更确定的结果。
        probs = torch.softmax(logits, dim=-1)
        # 生成与 probs 形状相同的张量，并就地填充从Exp(1)指数分布中采样的随机数，.clamp_min_(1e-10)防止除以零，之后就地计算除法，在词表维度找最大值的索引。本质是 Gumbel-Max Trick 的变体。
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
