# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA QSA owner with Triton kernels."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import ClassVar, cast

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)

logger = init_logger(__name__)
_FP8_E4M3_MAX = 448.0
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding, get_rope
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.utils.torch_utils import (
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

from ..common.qsa_cache import QSAForwardMetadata
from . import model
from .indexer_qsa import QSAIndexer


class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Flash metadata supporting uniform decode and target-verify graphs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class Qwen4ExpQSAFlashAttentionBackend(FlashAttentionBackend):
    """FullAttentionSpec backend used by the merged QSA owner."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    # fp8/fp8_e4m3: e4m3 bytes in a uint8 cache, written by reshape_and_cache
    # with the layer's per-tensor scales and dequantized on load inside the QSA
    # Triton kernel. flash-attn never runs over this cache, so its fp8 probe
    # does not apply (see supports_kv_cache_dtype and the impl constructor).
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype is None or kv_cache_dtype in cls.supported_kv_cache_dtypes

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        # QSA dequantizes the fp8 KV in its own Triton kernel and never runs
        # flash-attn over the quantized cache, so the parent's fp8-KV rejection
        # does not apply and every combination it is handed is accepted here.
        return None

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA_TRITON"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        # QSA consumes manager pages directly and does not use FA4 paged attention.
        return [MultipleOf(16)]

    @staticmethod
    def get_impl_cls() -> type[Qwen4ExpQSAFlashAttentionImpl]:
        return Qwen4ExpQSAFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Qwen4ExpQSAMetadataBuilder]:
        return Qwen4ExpQSAMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False



def _apply_qsa_kv_scales(
    layer: nn.Module, k_scale: float, v_scale: float, source: str
) -> None:
    """Write per-tensor FP8 KV scales onto the QSA owner (buffer + host floats)."""
    k_scale = float(k_scale)
    v_scale = float(v_scale)
    if hasattr(layer, "_k_scale"):
        layer._k_scale.fill_(k_scale)
    if hasattr(layer, "_v_scale"):
        layer._v_scale.fill_(v_scale)
    layer._k_scale_float = k_scale
    layer._v_scale_float = v_scale
    if hasattr(layer, "_k_scale_cpu"):
        layer._k_scale_cpu.fill_(k_scale)
    if hasattr(layer, "_v_scale_cpu"):
        layer._v_scale_cpu.fill_(v_scale)
    logger.info(
        "QSA FP8 KV scales on %s from %s: k_scale=%.6g v_scale=%.6g",
        getattr(layer, "layer_name", type(layer).__name__),
        source,
        k_scale,
        v_scale,
    )


def _sync_qsa_kv_scale_floats(layer: nn.Module) -> None:
    """Keep host floats in sync with the persistent scale buffers after load."""
    if hasattr(layer, "_k_scale"):
        layer._k_scale_float = float(layer._k_scale.item())
        if hasattr(layer, "_k_scale_cpu"):
            layer._k_scale_cpu.fill_(layer._k_scale_float)
    if hasattr(layer, "_v_scale"):
        layer._v_scale_float = float(layer._v_scale.item())
        if hasattr(layer, "_v_scale_cpu"):
            layer._v_scale_cpu.fill_(layer._v_scale_float)


def _maybe_load_qsa_kv_scales_file(layer: nn.Module) -> bool:
    """Load scales from VLLM_QSA_KV_SCALES_PATH JSON if present."""
    path = os.environ.get("VLLM_QSA_KV_SCALES_PATH", "").strip()
    if not path or not os.path.isfile(path):
        return False
    data = json.loads(Path(path).read_text())
    layers = data.get("layers", data)
    layer_id = getattr(layer, "_qsa_layer_id", getattr(layer, "layer_idx", None))
    keys: list[str] = []
    if layer_id is not None:
        keys.append(str(layer_id))
    name = getattr(layer, "layer_name", None)
    if name:
        keys.append(name)
    entry = None
    for k in keys:
        if k in layers:
            entry = layers[k]
            break
    if entry is None:
        return False
    if "k_scale" in entry and "v_scale" in entry:
        k_scale, v_scale = float(entry["k_scale"]), float(entry["v_scale"])
    elif "k_amax" in entry and "v_amax" in entry:
        k_scale = max(float(entry["k_amax"]), 1e-6) / _FP8_E4M3_MAX
        v_scale = max(float(entry["v_amax"]), 1e-6) / _FP8_E4M3_MAX
    else:
        return False
    _apply_qsa_kv_scales(layer, k_scale, v_scale, path)
    return True


def _finalize_qsa_kv_calib(layer: nn.Module) -> None:
    k_amax = float(getattr(layer, "_qsa_k_amax", 0.0))
    v_amax = float(getattr(layer, "_qsa_v_amax", 0.0))
    margin = float(os.environ.get("VLLM_QSA_KV_CALIB_MARGIN", "1.25") or "1.25")
    k_scale = max(k_amax * margin, 1e-6) / _FP8_E4M3_MAX
    v_scale = max(v_amax * margin, 1e-6) / _FP8_E4M3_MAX
    _apply_qsa_kv_scales(layer, k_scale, v_scale, "online-calib")
    out = os.environ.get("VLLM_QSA_KV_CALIB_OUT", "").strip()
    if not out:
        return
    layer_id = getattr(layer, "_qsa_layer_id", getattr(layer, "layer_idx", None))
    key = str(
        layer_id if layer_id is not None else getattr(layer, "layer_name", "unknown")
    )
    entry = {
        "k_amax": k_amax,
        "v_amax": v_amax,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "layer_name": getattr(layer, "layer_name", None),
    }

    def _write() -> None:
        data: dict = {"layers": {}}
        if Path(out).is_file():
            try:
                data = json.loads(Path(out).read_text())
                data.setdefault("layers", {})
            except Exception:
                data = {"layers": {}}
        data["layers"][key] = entry
        data["fp8_e4m3_max"] = _FP8_E4M3_MAX
        Path(out).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    lock_path = out + ".lock"
    with open(lock_path, "a+", encoding="utf-8") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            _write()
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    logger.info("QSA FP8 KV calib wrote layer %s -> %s", key, out)


class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
    """Run paged sparse GQA with the QSA Triton kernel."""

    supports_dcp: bool = False

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # Optional online absmax collection for offline FP8 KV scale files.
        remaining = getattr(layer, "_qsa_kv_calib_remaining", 0)
        warmup = int(getattr(layer, "_qsa_kv_calib_warmup_remaining", 0))
        # .item() syncs would invalidate CUDA graph capture.
        if (remaining > 0 or warmup > 0) and key.numel() > 0 and not torch.cuda.is_current_stream_capturing():
            n_all = min(int(slot_mapping.numel()), int(key.shape[0]))
            offset = 0
            if warmup > 0:
                take = min(n_all, warmup)
                layer._qsa_kv_calib_warmup_remaining = warmup - take
                offset = take
            usable = n_all - offset
            if remaining > 0 and usable > 0:
                n = min(usable, remaining)
                sl = slice(offset, offset + n)
                k_abs = key[sl].detach().float().abs().amax().item()
                v_abs = value[sl].detach().float().abs().amax().item()
                layer._qsa_k_amax = max(float(getattr(layer, "_qsa_k_amax", 0.0)), k_abs)
                layer._qsa_v_amax = max(float(getattr(layer, "_qsa_v_amax", 0.0)), v_abs)
                layer._qsa_kv_calib_remaining = remaining - n
                if layer._qsa_kv_calib_remaining <= 0:
                    _finalize_qsa_kv_calib(layer)
        super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

    supports_pcp: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        # The parent constructor probes flash-attn for quantized-KV support and
        # raises where it is unavailable (sm120), but QSA dequantizes fp8 inside
        # its own Triton kernel and never runs flash-attn over the cache. Hand
        # the parent "auto" for that probe and restore the real dtype afterwards:
        # the parent only uses it there, and do_kv_cache_update reads the
        # attribute at call time.
        real_kv_cache_dtype = kv_cache_dtype
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            kv_cache_dtype = "auto"
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            sinks,
        )
        self.kv_cache_dtype = real_kv_cache_dtype
        if not is_flash_attn_varlen_func_available():
            raise NotImplementedError("Qwen4Exp QSA requires FlashAttention")
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                "Qwen4Exp QSA does not support decode context parallelism"
            )
        if self.kv_cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16 or FP8-e4m3 main KV cache"
            )
        self.supports_quant_query_input = False

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        token_to_req: torch.Tensor,
        use_prefill_config: bool,
        output_gate: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA does not support fused output quantization")
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("QSA does not support ALiBi or attention sinks")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError("QSA does not support sliding-window attention")

        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output

        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        logical_indices = topk_buffer[:num_tokens]
        token_to_req = token_to_req[:num_tokens]
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        k_scale = v_scale = None
        if self.kv_cache_dtype in ("fp8", "fp8_e4m3"):
            # The cache is allocated as uint8; reinterpret the e4m3 bytes
            # (same itemsize, so shape and strides are preserved).
            key_cache = key_cache.view(torch.float8_e4m3fn)
            value_cache = value_cache.view(torch.float8_e4m3fn)
            # Host-side per-tensor dequant scales (Python floats), as used by
            # other host-scale backends; folded into the kernel's scales.
            k_scale = layer._k_scale_float
            v_scale = layer._v_scale_float
        if query.dtype != torch.bfloat16 or key_cache.dtype not in (
            torch.bfloat16,
            torch.float8_e4m3fn,
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires BF16 Q and BF16 or FP8-e4m3 K/V"
            )

        from .ops.qsa import qsa_sparse_paged_attention

        qsa_sparse_paged_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            use_prefill_config,
            output[:num_tokens],
            k_scale=k_scale,
            v_scale=v_scale,
            output_gate=output_gate[:num_tokens],
        )
        return output


class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch."""

    supports_dcp = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen4Exp QSA requires a paged KV cache")
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA currently requires BF16")
        if cache_config.cache_dtype not in (
            "auto",
            "bfloat16",
            "fp8",
            "fp8_e4m3",
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16 or FP8-e4m3 main KV cache"
            )
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support KV quantization")
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA does not support context parallelism"
            )
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("Qwen4Exp QSA requires causal decoder attention")

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        # Decode/verify batches have at most 1 + num_spec query tokens per
        # request; use_prefill_config (max_query_len > this) steers the
        # config table. Shorter batches take the decode profile — harmless,
        # the difference is tile-shape tuning, not correctness.
        self._max_decode_query_len = 1 + vllm_config.num_speculative_tokens
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support dual-chunk RoPE")
        # Qwen4Exp full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=model.without_modelopt_fp4(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        mm_config = model_config.multimodal_config
        text_only = mm_config is None or mm_config.language_model_only
        mrope_section = getattr(self.rotary_emb, "mrope_section", None)
        supports_mrope = bool(
            type(self.rotary_emb) is MRotaryEmbedding
            and mrope_section
            and len(mrope_section) == 3
            and sum(mrope_section) == self.rotary_emb.rotary_dim // 2
            and getattr(self.rotary_emb, "mrope_interleaved", False)
        )
        supports_dtype = getattr(self.rotary_emb, "dtype", None) in (
            torch.float16,
            torch.bfloat16,
        )
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and supports_dtype
            and (text_only or supports_mrope)
        )

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = cache_config.cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        if self.kv_cache_torch_dtype not in (torch.bfloat16, torch.uint8):
            raise NotImplementedError(
                "Qwen4Exp QSA requires BF16 or FP8-e4m3 (uint8) cache storage"
            )
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)
        self._qsa_layer_id = int(layer_id)
        # Prefer checkpoint-loaded buffers; optional JSON overrides (calib output).
        if not _maybe_load_qsa_kv_scales_file(self):
            _sync_qsa_kv_scale_floats(self)
        calib_tokens = int(os.environ.get("VLLM_QSA_KV_CALIB_TOKENS", "0") or "0")
        if calib_tokens > 0 and os.environ.get("VLLM_QSA_KV_CALIB_OUT", "").strip():
            self._qsa_kv_calib_remaining = calib_tokens
            self._qsa_kv_calib_warmup_remaining = int(
                os.environ.get("VLLM_QSA_KV_CALIB_WARMUP_TOKENS", "4096") or "0"
            )
            self._qsa_k_amax = 0.0
            self._qsa_v_amax = 0.0
            logger.info(
                "QSA FP8 KV calib armed on layer %s for %d tokens "
                "(skip first %d warmup) -> %s",
                self.layer_name,
                calib_tokens,
                self._qsa_kv_calib_warmup_remaining,
                os.environ.get("VLLM_QSA_KV_CALIB_OUT"),
            )
        elif self.kv_cache_dtype in ("fp8", "fp8_e4m3"):
            logger.info(
                "QSA FP8 KV scales on %s: k_scale=%.6g v_scale=%.6g",
                self.layer_name,
                float(self._k_scale_float),
                float(self._v_scale_float),
            )

        self.attn_backend = Qwen4ExpQSAFlashAttentionBackend
        self.impl = Qwen4ExpQSAFlashAttentionImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            None,
            None,
            self.kv_cache_dtype,
            None,
            AttentionType.DECODER,
            None,
        )
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        # PACKED selection buffer: the trailing column holds each row's
        # valid-entry count (written by the expand kernel) — never a token
        # index; the sparse attention kernel reads it as its loop bound.
        # MTP skip_topk steps reuse rows frozen from step 0; the count is
        # a row column, so compaction/reuse keep it paired with the content.
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(
                max_tokens,
                self.indexer.packed_output_width,
                dtype=torch.int32,
            ),
            persistent=False,
        )

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def process_weights_after_loading(self, act_dtype: torch.dtype | None = None) -> None:
        # Checkpoint may have filled _k_scale/_v_scale buffers via
        # _remap_qsa_cache_scale_name; host floats are not auto-synced unless
        # BaseKVCacheMethod runs (QSA does not use that path).
        if os.environ.get("VLLM_QSA_KV_SCALES_PATH", "").strip():
            _maybe_load_qsa_kv_scales_file(self)
        else:
            _sync_qsa_kv_scale_floats(self)
        if self.kv_cache_dtype in ("fp8", "fp8_e4m3"):
            logger.info(
                "QSA FP8 KV scales after weight load on %s: k_scale=%.6g v_scale=%.6g",
                self.layer_name,
                float(self._k_scale_float),
                float(self._v_scale_float),
            )
        parent = super()
        if hasattr(parent, "process_weights_after_loading"):
            try:
                parent.process_weights_after_loading(act_dtype)  # type: ignore[misc]
            except TypeError:
                parent.process_weights_after_loading()  # type: ignore[misc]


    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    @eager_break_during_capture
    def _run_qsa(
        self,
        projected_qk: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        output_gate: torch.Tensor,
    ) -> None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            return
        main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(
            QSAForwardMetadata,
            metadata[self.indexer.raw_key_cache.prefix],
        )
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        selected = self.indexer(
            projected_qk,
            positions,
            self.topk_indices_buffer[:num_tokens],
        )
        if selected.shape != (num_tokens, self.indexer.packed_output_width):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        impl.do_kv_cache_update(
            self,
            key,
            value,
            self.kv_cache,
            main_metadata.slot_mapping,
        )
        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
            use_prefill_config=main_metadata.max_query_len > self._max_decode_query_len,
            output_gate=output_gate,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        assert gate is not None
        num_tokens = hidden_states.shape[0]
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        attn_output = torch.empty_like(query)
        # Keep the index projection outside the eager break.
        projected_qk, _ = self.indexer.index_qk_proj(hidden_states)
        self._run_qsa(
            projected_qk,
            positions,
            query,
            key,
            value,
            attn_output,
            gate,
        )
        flat_output = attn_output.view(num_tokens, -1)
        output, _ = self.o_proj(flat_output)
        return output


__all__ = [
    "QSAIndexer",
    "Qwen4ExpQSAAttention",
    "Qwen4ExpQSAFlashAttentionBackend",
    "Qwen4ExpQSAFlashAttentionImpl",
]
