import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.activations import ACT2FN


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


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """将 [batch, seq, kv_heads, dim] 的每个 KV 头分配给 n_rep 个 Q 头。"""
    return x if n_rep == 1 else x.repeat_interleave(n_rep, dim=2)


class Attention(nn.Module):
    """带 RoPE 的因果 GQA，输入和输出均为 [batch, seq, hidden_size]。"""

    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        if self.n_heads <= 0 or self.n_kv_heads <= 0:
            raise ValueError("Attention head counts must be positive")
        if config.hidden_size % self.n_heads or self.n_heads % self.n_kv_heads:
            raise ValueError("hidden_size must divide evenly into Q heads, and Q heads into KV groups")
        self.head_dim = config.hidden_size // self.n_heads
        if self.head_dim <= 0 or self.head_dim % 2:
            raise ValueError("RoPE requires a positive even head_dim")
        self.n_rep = self.n_heads // self.n_kv_heads
        self.flash = config.flash_attention
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """position_embeddings 传完整 cos/sin 表，本层按缓存长度自动切片。

        缓存为未重复的 (旋转后 K, V)，形状 [batch, past_seq, kv_heads, dim]。
        attention_mask 可选，形状 [batch, past_seq + seq]，1/True 表示有效 key。
        返回 (输出, 新缓存或 None)。批内共享位置编号，不单独压缩 padding 位置。
        """
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        past_len = 0
        if past_key_value is not None:
            pk, pv = past_key_value
            if (pk.ndim != 4 or pk.shape != pv.shape or pk.shape[0] != batch
                    or pk.shape[2:] != (self.n_kv_heads, self.head_dim)):
                raise ValueError("Invalid KV cache shape")
            past_len = pk.shape[1]

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(
            q, k, cos[past_len:past_len + seq_len], sin[past_len:past_len + seq_len]
        )
        if past_key_value is not None:
            k = torch.cat((pk, k), dim=1)
            v = torch.cat((pv, v), dim=1)
        present = (k, v) if use_cache else None

        q = q.transpose(1, 2)
        k = repeat_kv(k, self.n_rep).transpose(1, 2)
        v = repeat_kv(v, self.n_rep).transpose(1, 2)
        total_len = past_len + seq_len
        # 缓存解码时 query 的绝对位置从 past_len 开始。
        allowed = None
        if past_len or attention_mask is not None or not self.flash:
            queries = torch.arange(seq_len, device=x.device) + past_len
            keys = torch.arange(total_len, device=x.device)
            allowed = (keys[None, :] <= queries[:, None])[None, None, :, :]
            if attention_mask is not None:
                if attention_mask.shape != (batch, total_len):
                    raise ValueError("attention_mask must cover both cached and current tokens")
                allowed = allowed & attention_mask.to(device=x.device, dtype=torch.bool)[:, None, None, :]

        if self.flash:
            # PyTorch 根据设备选择可用后端，不保证一定使用 FlashAttention 内核。
            output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=allowed,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=allowed is None,
            )
        else:
            scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~allowed, float("-inf"))
            # 全被遮挡的 padding 行输出零，避免 softmax(-inf) 产生 NaN。
            has_keys = allowed.any(dim=-1, keepdim=True)
            scores = scores.masked_fill(~has_keys, 0.0)
            probs = scores.softmax(dim=-1).masked_fill(~has_keys, 0.0).to(v.dtype)
            output = F.dropout(probs, p=self.dropout, training=self.training) @ v

        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.resid_dropout(self.o_proj(output)), present


class FeedForward(nn.Module):
    """门控前馈网络；hidden_act='silu' 时为 SwiGLU。"""

    def __init__(self, config: MokioMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            # 三个投影采用约 8/3 倍宽度，并向上对齐到 64 的倍数。
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 63) // 64)
        if config.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, seq, hidden] -> [batch, seq, hidden]，分别处理每个 token。"""
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))


class MiniMindBlock(nn.Module):
    """Pre-norm Transformer 层，Attention 和 FFN 各带一条残差连接。"""

    def __init__(self, layer_id: int, config: MokioMindConfig):
        super().__init__()
        if config.use_moe:
            raise NotImplementedError("MOEFeedForward is not implemented; use use_moe=False")
        self.layer_id = layer_id
        self.self_attn = Attention(config)
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.self_attn.head_dim
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """传入完整 cos/sin 表及本层缓存；输出形状保持 [batch, seq, hidden]。"""
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """Embedding、多个 Transformer Block 和最终 RMSNorm，不包含词表输出层。"""

    def __init__(self, config: MokioMindConfig):
        super().__init__()
        if config.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [MiniMindBlock(layer_id, config) for layer_id in range(self.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        cos, sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list | tuple | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list, torch.Tensor]:
        """返回 (hidden_states, 每层缓存列表, aux_loss)。

        input_ids: [batch, 当前序列长度]；mask 覆盖历史和当前全部 key。
        缓存仅支持每层 (K, V) 的列表/元组；不支持框架 Cache 对象。
        无缓存输出时列表内为 None；当前普通 FFN 的 aux_loss 为零。
        """
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [batch, nonempty seq]")
        if past_key_values is None:
            past_key_values = [None] * self.num_hidden_layers
        elif not isinstance(past_key_values, (list, tuple)):
            raise TypeError("past_key_values must be a list/tuple of per-layer (K, V) caches")
        if len(past_key_values) != self.num_hidden_layers:
            raise ValueError("past_key_values must contain exactly one entry per layer")

        lengths = []
        for past in past_key_values:
            if past is None:
                lengths.append(0)
                continue
            if not isinstance(past, (list, tuple)) or len(past) != 2:
                raise ValueError("Each layer cache must be a (K, V) pair")
            k, v = past
            expected_tail = (self.config.num_key_value_heads, self.layers[0].head_dim)
            if (not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor)
                    or k.ndim != 4 or k.shape != v.shape
                    or k.shape[0] != input_ids.shape[0] or k.shape[2:] != expected_tail):
                raise ValueError("Invalid layer cache shape")
            lengths.append(k.shape[1])
        if len(set(lengths)) != 1:
            raise ValueError("All layer caches must have the same sequence length")
        total_length = lengths[0] + input_ids.shape[1]
        if total_length > self.freqs_cos.shape[0]:
            raise ValueError("Sequence including cache exceeds max_position_embeddings")
        if attention_mask is not None and attention_mask.shape != (input_ids.shape[0], total_length):
            raise ValueError("attention_mask must cover cached and current tokens")

        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Attention 内部按缓存长度切片，这里必须传完整位置表。
        position_embeddings = (self.freqs_cos, self.freqs_sin)
        presents = []
        for layer, past in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = hidden_states.new_zeros(())
        return hidden_states, presents, aux_loss


class MokioMindForCausalLM(PreTrainedModel, GenerationMixin):
    """在骨干模型上加入共享权重的词表投影和 next-token loss。"""

    config_class = MokioMindConfig
    base_model_prefix = "model"
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MokioMindConfig):
        config.tie_word_embeddings = True
        super().__init__(config)
        self.model = MiniMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, MiniMindModel):
            # from_pretrained 的 meta 初始化后需重建非持久化 buffer。
            cos, sin = precompute_freqs_cis(
                dim=self.config.hidden_size // self.config.num_attention_heads,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling,
            )
            module.freqs_cos = cos.to(module.freqs_cos.device)
            module.freqs_sin = sin.to(module.freqs_sin.device)

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=True, **kwargs
    ):
        past_len = 0
        if isinstance(past_key_values, DynamicCache):
            past_len = past_key_values.get_seq_length()
        elif past_key_values:
            past_len = past_key_values[0][0].shape[1]
        if past_len:
            input_ids = input_ids[:, past_len:]
        return dict(
            input_ids=input_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, use_cache=use_cache, logits_to_keep=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ):
        # Transformers 的 DynamicCache 使用 [batch, heads, seq, dim] 布局。
        dynamic_cache = past_key_values if isinstance(past_key_values, DynamicCache) else None
        past_len = dynamic_cache.get_seq_length() if dynamic_cache is not None else 0
        legacy_cache = past_key_values
        if dynamic_cache is not None:
            legacy_cache = None if past_len == 0 else [
                (layer.keys.transpose(1, 2), layer.values.transpose(1, 2))
                for layer in dynamic_cache.layers
            ]
        hidden_states, presents, aux_loss = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            past_key_values=legacy_cache, use_cache=use_cache,
        )
        if isinstance(logits_to_keep, int):
            if logits_to_keep < 0:
                raise ValueError("logits_to_keep must be nonnegative")
            indices = slice(-logits_to_keep, None)
        else:
            indices = logits_to_keep

        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must match input_ids shape")
            # loss 始终使用完整当前序列，避免 logits 截断后与 labels 错位。
            full_logits = self.lm_head(hidden_states)
            targets = labels[:, 1:].to(full_logits.device)
            shifted = full_logits[:, :-1].float()
            if targets.numel() and (targets != -100).any():
                loss = F.cross_entropy(
                    shifted.reshape(-1, self.config.vocab_size), targets.reshape(-1), ignore_index=-100
                )
            else:
                loss = full_logits.sum() * 0.0
            logits = full_logits[:, indices, :]
        else:
            logits = self.lm_head(hidden_states[:, indices, :])

        output_cache = None
        if use_cache:
            if dynamic_cache is None:
                dynamic_cache = DynamicCache(config=self.config)
                past_len = 0
            for layer_idx, (k, v) in enumerate(presents):
                dynamic_cache.update(
                    k[:, past_len:].transpose(1, 2), v[:, past_len:].transpose(1, 2), layer_idx
                )
            output_cache = dynamic_cache
        output = CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=output_cache,
            hidden_states=(hidden_states,),
        )
        output["aux_loss"] = aux_loss
        return output
