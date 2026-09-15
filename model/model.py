import math

import torch
from torch import nn
from transformers import PretrainedConfig


class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )
class RMSNorm(nn.Module):
    def __init__(self,dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return  torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * self._norm(x)


def precompute_freqs_cis(
    dim: int,
    end: int = 32 * 1024,
    rope_base: float = 1000000.0,
    rope_scaling: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """预计算 RoPE 的 cos/sin 表，形状均为 [end, dim]。

    dim 是每个注意力头的维度。采用前后半维配对的旋转布局；
    rope_scaling=None 使用普通 RoPE，否则按 YaRN 调整频率。
    """
    if dim <= 0 or dim % 2:
        raise ValueError("dim must be a positive even integer")
    if end <= 0 or rope_base <= 1:
        raise ValueError("end must be positive and rope_base must exceed 1")

    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))
    attention_factor = 1.0

    if rope_scaling is not None:
        if rope_scaling.get("rope_type", rope_scaling.get("type", "yarn")) != "yarn":
            raise ValueError("Only yarn rope scaling is supported")
        factor = float(rope_scaling.get("factor", 1.0))
        original_length = int(rope_scaling["original_max_position_embeddings"])
        beta_fast = float(rope_scaling.get("beta_fast", 32))
        beta_slow = float(rope_scaling.get("beta_slow", 1))
        if factor < 1 or original_length <= 0 or not 0 < beta_slow < beta_fast:
            raise ValueError("Invalid YaRN factor, original length or beta range")

        def correction_dim(rotations: float) -> float:
            return dim * math.log(original_length / (2 * math.pi * rotations)) / (
                2 * math.log(rope_base)
            )

        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = ((torch.arange(dim // 2).float() - low) / (high - low)).clamp(0, 1)
        # 高频保留原频率，低频除以扩展倍数，中频平滑过渡。
        freqs = freqs * (1 - ramp) + (freqs / factor) * ramp
        attention_factor = rope_scaling.get("attention_factor")
        if attention_factor is None:
            attention_factor = 1.0 + 0.1 * math.log(factor)

    angles = torch.outer(torch.arange(end).float(), freqs)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos() * attention_factor, angles.sin() * attention_factor


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """旋转 Q/K：[batch, seq_len, heads, head_dim]，允许 Q/K 头数不同。

    cos/sin 为当前序列位置对应的 [seq_len, head_dim] 切片。
    KV cache 解码时，调用方需按历史长度切片，例如 cos[start:start+seq_len]。
    """
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must have shape [batch, seq_len, heads, head_dim]")
    if q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1] or q.shape[-1] % 2:
        raise ValueError("q and k must share batch, sequence and even head dimensions")
    if cos.shape != (q.shape[1], q.shape[-1]) or sin.shape != cos.shape:
        raise ValueError("cos and sin must have shape [seq_len, head_dim]")

    def rotate(x: torch.Tensor) -> torch.Tensor:
        first, second = x.float().chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        c = cos.to(device=x.device, dtype=torch.float32)[None, :, None, :]
        s = sin.to(device=x.device, dtype=torch.float32)[None, :, None, :]
        return (x.float() * c + rotated * s).to(x.dtype)

    return rotate(q), rotate(k)
