from __future__ import annotations

import math
from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("OpenAIPrivacyFilterForTokenClassification")
class PrivacyFilterModel(TextModel):
    model_arch = gguf.MODEL_ARCH.PRIVACY_FILTER

    def _yarn_inv_freq(self) -> tuple[Tensor, Tensor, float]:
        rope = self.hparams["rope_parameters"]

        dim       = self.hparams["head_dim"]
        base      = rope["rope_theta"]
        factor    = rope["factor"]
        orig_ctx  = rope["original_max_position_embeddings"]
        beta_fast = rope.get("beta_fast") or 32.0
        beta_slow = rope.get("beta_slow") or 1.0

        # this model sets truncate=false, whereas ggml_rope_yarn_corr_dims() always
        # rounds the correction range, so the ramp cannot be expressed via GGUF hparams
        truncate = rope.get("truncate", True)

        def corr_dim(n_rot: float) -> float:
            return (dim * math.log(orig_ctx / (n_rot * 2 * math.pi))) / (2 * math.log(base))

        low, high = corr_dim(beta_fast), corr_dim(beta_slow)
        if truncate:
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, dim - 1)
        if low == high:
            high += 0.001

        pos_freqs  = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        inv_extrap = 1.0 / pos_freqs
        inv_interp = 1.0 / (factor * pos_freqs)

        ramp   = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
        extrap = 1.0 - ramp

        inv_freq = inv_interp * (1 - extrap) + inv_extrap * extrap

        attn_factor = 1.0 if factor <= 1.0 else 0.1 * math.log(factor) + 1.0

        return inv_extrap, inv_freq, attn_factor

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        inv_extrap, inv_freq, _ = self._yarn_inv_freq()

        # ggml computes theta = theta_extrap / freq_factor, so fold the YaRN ramp in here
        # and let the rope op run unscaled
        yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS), inv_extrap / inv_freq)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if "sinks" in name:
            name += ".weight"

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if "down_proj" in name:
            if name.endswith("_bias"):
                name = name.replace("down_proj_bias", "down_proj.bias")
            else:
                name = name.replace("down_proj", "down_proj.weight")
                data_torch = data_torch.transpose(-1, -2)

        # unlike gpt-oss, gate and up are concatenated halves rather than interleaved
        if "gate_up_proj" in name:
            if name.endswith("_bias"):
                name_gate = name.replace("gate_up_proj_bias", "gate_proj.bias")
                name_up   = name.replace("gate_up_proj_bias", "up_proj.bias")
                gate_b, up_b = data_torch.chunk(2, dim=-1)
                yield from super().modify_tensors(gate_b, name_gate, bid)
                yield from super().modify_tensors(up_b, name_up, bid)
                return

            name_gate = name.replace("gate_up_proj", "gate_proj.weight")
            name_up   = name.replace("gate_up_proj", "up_proj.weight")
            data_torch = data_torch.transpose(-1, -2)
            gate_w, up_w = data_torch.chunk(2, dim=-2)
            yield from super().modify_tensors(gate_w, name_gate, bid)
            yield from super().modify_tensors(up_w, name_up, bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        _, _, attn_factor = self._yarn_inv_freq()

        # the YaRN ramp is baked into the rope_freqs tensor, so emit no rope scaling
        # and apply the magnitude scaling that rope_yarn() would otherwise derive
        self.rope_parameters = {k: v for k, v in self.rope_parameters.items() if k != "rope_type"}

        super().set_gguf_parameters()

        self.gguf_writer.add_rope_scaling_attn_factors(attn_factor)

        self.gguf_writer.add_expert_feed_forward_length(self.hparams["intermediate_size"])

        # the model attends to abs(i - j) <= sliding_window, and LLAMA_SWA_TYPE_SYMMETRIC
        # halves this key, so store the full window width
        self.gguf_writer.add_sliding_window(2 * self.hparams["sliding_window"])

        self.gguf_writer.add_causal_attention(False)
        self.gguf_writer.add_pooling_type(gguf.PoolingType.NONE)

        id2label = {int(k): v for k, v in self.hparams["id2label"].items()}
        labels = [id2label[i] for i in range(len(id2label))]
        self.gguf_writer.add_classifier_output_labels(labels)
        self.gguf_writer.add_embedding_length_out(len(labels))
