# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

# isort: off
# This verifies aimet is installed, and this must be included first.
from qai_hub_models.models._shared.llm.model import (
    LLMBase,
    LLM_AIMETOnnx,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
)

# isort: on
import copy
import json
import os
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import onnx
import torch

if TYPE_CHECKING:
    from aimet_onnx.quantsim import QuantizationSimModel

from packaging.version import Version
from transformers import PretrainedConfig, PreTrainedTokenizer
from transformers.models.llama import LlamaConfig
from transformers.models.exaone4.modeling_exaone4 import Exaone4RotaryEmbedding

from qai_hub_models.models._shared.exaone4.model_adaptations import (
    QcExaone4_apply_rotary_pos_emb,
    QCExaone4ForCausalLM,
    QCExaone4MLP,
    QCExaone4Attention,
    QCExaone4DecoderLayer,
)
from qai_hub_models.models._shared.llm.model import Embedding, PositionProcessorBase
from qai_hub_models.utils.aimet.encodings import propagate_memory_encodings

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# Configs
AIMET_ENCODINGS_PREFIX = "config"
AIMET_CONFIG = "default_config_exaone4"

DATA_DIR = "data"
USE_CACHED_DATA = True

# EXAONE4 special tokens and templates
BEGIN_ROLE = "[|{role}|]\n"
END_TURN = "[|endofturn|]\n"
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
SYSTEM_ROLE = "system"
TOOL_ROLE = "tool"
THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
TOOL_RESULT_START = "<tool_result>"
TOOL_RESULT_END = "</tool_result>"

DEFAULT_PROMPT_CONTEXT = "You are a helpful AI assistant"
DEFAULT_USER_PROMPT = "What are the key features of EXAONE 4.0? Keep the answer under ten words."


class Exaone4_Optimizations(str, Enum):
    """Enum for EXAONE4-specific optimizations"""
    SHA_ATTENTION = "sha_attention"
    RMS_NORM_4_RANK = "rank4_rms_norm"
    QK_NORM = "qk_norm"
    HYBRID_ATTENTION = "hybrid_attention"


class RopeEmbedding(Embedding):
    """RoPE embedding for EXAONE4"""

    def __init__(
        self,
        head_dim: int = 64,
        max_length: int = 2048,
        config: LlamaConfig = LlamaConfig(),
    ) -> None:
        self.cos, self.sin = self.precompute(head_dim, max_length, config)

    def precompute(
        self, head_dim: int, max_length: int, config: LlamaConfig
    ) -> list[torch.Tensor]:
        """Precompute RoPE embeddings"""
        from transformers.models.exaone4 import modeling_exaone4

        head_dim = (
            config.head_dim
            if hasattr(config, "head_dim")
            else config.hidden_size // config.num_attention_heads
        )

        if not hasattr(config, "rope_scaling"):
            setattr(config, "rope_scaling", None)

        # Create EXAONE4 RoPE embeddings for proper attention scaling
        rope = modeling_exaone4.Exaone4RotaryEmbedding(config)
        dummy_x = torch.tensor([1.0])
        position_ids = torch.arange(max_length).view(1, -1)

        if hasattr(rope, "_original_forward"):
            embeddings = rope._original_forward(dummy_x, position_ids)
        else:
            embeddings = rope.forward(dummy_x, position_ids)

        # Adjust for EXAONE4 head dimension
        emb_size = embeddings[0].size(-1) // 2
        embeddings = [emb[:, :, :emb_size] for emb in embeddings]
        embeddings = [emb.unsqueeze(0) for emb in embeddings]
        return embeddings  # pyright: ignore [reportReturnType]

    def get_embedding(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get RoPE embeddings for given position IDs
        position_ids: [batch_size, sequence_length]
        return [batch_size, 1, sequence_length, head_dim//2][2]
        """
        cos = self.cos[0, 0, :, :].to(position_ids.device)  # [seq_len, dim]
        sin = self.sin[0, 0, :, :].to(position_ids.device)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1).to(dtype=dtype)
        sin = sin[position_ids].unsqueeze(1).to(dtype=dtype)
        return cos, sin


class Exaone4PositionProcessor(PositionProcessorBase):
    """
    Position processor for EXAONE4 models
    """

    def __init__(self, context_length: int):
        super().__init__(context_length)
        self.context_len = context_length
        self.rope_embedding = RopeEmbedding(max_length=self.context_len)

    def forward(self, attention_mask_before_processor, position_ids):
        from qai_hub_models.models._shared.llm.model import (
            prepare_combined_attention_mask,
        )

        position_ids_cos, position_ids_sin = self.rope_embedding.get_embedding(
            position_ids
        )
        attention_mask = prepare_combined_attention_mask(
            attention_mask_before_processor,
            position_ids.shape,
            attention_mask_before_processor.shape[1] - position_ids.shape[1],
        )
        return attention_mask, position_ids_cos, position_ids_sin


class Exaone4Base(LLMBase):
    """Base class for EXAONE4 models"""
    
    LMClass = QCExaone4ForCausalLM
    EmbeddingClass = RopeEmbedding

    @staticmethod
    def get_input_prompt_with_tags(
        user_input_prompt: str = DEFAULT_USER_PROMPT,
        system_context_prompt: str = DEFAULT_PROMPT_CONTEXT,
    ) -> str:
        """
        Get EXAONE4-formatted prompt with role tags
        """
        prompt = f"""{BEGIN_ROLE.format(role=SYSTEM_ROLE)}{system_context_prompt}{END_TURN}{BEGIN_ROLE.format(role=USER_ROLE)}{user_input_prompt}{END_TURN}{BEGIN_ROLE.format(role=ASSISTANT_ROLE)}{THINK_START}

{THINK_END}

"""
        return prompt

    @staticmethod
    def monkey_patch(
        skip_optimizations: list[str] | None = None,
    ) -> None:
        """Apply EXAONE4-specific monkey patches"""
        # Import EXAONE4 modeling modules
        try:
            from transformers.models.exaone4 import modeling_exaone4
        except ImportError as e:
            raise ImportError("EXAONE4 transformers not available. Please install the required package.") from e

        # Apply SHA attention optimization
        if (
            skip_optimizations
            and Exaone4_Optimizations.SHA_ATTENTION in skip_optimizations
        ):
            print("Skip sha_attention optimization")
        else:
            # Replace standard attention with optimized version
            modeling_exaone4.Exaone4Attention = QCExaone4Attention

        # Apply RoPE optimization
        def bypass_RotaryEmbedding(self, x, position_ids, *args, **kwargs):
            return position_ids

        if not hasattr(modeling_exaone4.Exaone4RotaryEmbedding, "_original_forward"):
            modeling_exaone4.Exaone4RotaryEmbedding._original_forward = (
                modeling_exaone4.Exaone4RotaryEmbedding.forward
            )
            modeling_exaone4.Exaone4RotaryEmbedding.forward = bypass_RotaryEmbedding

        # Apply optimized RoPE function
        from transformers.models.exaone4 import modeling_exaone4
        modeling_exaone4.apply_rotary_pos_emb = QcExaone4_apply_rotary_pos_emb

        # Apply rank-4 RMS norm optimization
        if (
            skip_optimizations
            and Exaone4_Optimizations.RMS_NORM_4_RANK in skip_optimizations
        ):
            print("Skip rank4_rms_norm optimization")
        else:
            def Exaone4RMSNorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                # Handle both 3D and 4D tensors (Exaone4 has q_norm/k_norm on 4D tensors)
                if hidden_states.dim() == 3:
                    # Raise to rank 4 for better performance on Qualcomm hardware (3D -> 4D)
                    hidden_states = hidden_states.unsqueeze(0)
                    variance = hidden_states.pow(2).mean(-1, keepdim=True)
                    hidden_states = hidden_states * torch.rsqrt(
                        variance + self.variance_epsilon
                    )
                    return (hidden_states * self.weight).squeeze(0)
                else:
                    # For 4D tensors (like q_norm/k_norm), use standard RMS norm
                    variance = hidden_states.pow(2).mean(-1, keepdim=True)
                    hidden_states = hidden_states * torch.rsqrt(
                        variance + self.variance_epsilon
                    )
                    return hidden_states * self.weight

            modeling_exaone4.Exaone4RMSNorm.forward = Exaone4RMSNorm_forward

        # Replace MLP and main model classes
        modeling_exaone4.Exaone4MLP = QCExaone4MLP
        modeling_exaone4.Exaone4ForCausalLM = QCExaone4ForCausalLM
        modeling_exaone4.Exaone4DecoderLayer = QCExaone4DecoderLayer

    def _verify_ckpt(self):
        """Verify checkpoint compatibility"""
        if not (
            self.llm_config.architectures[0] == "Exaone4ForCausalLM"
            and self.llm_config.model_type == "exaone4"
        ):
            raise ValueError(
                "Model config is not compatible with this EXAONE4 implementation."
            )

class Exaone4Base_AIMETOnnx(LLM_AIMETOnnx):
    """AIMET ONNX base class for EXAONE4 models"""
    
    EmbeddingClass = RopeEmbedding

    def __init__(
        self,
        sim_model: QuantizationSimModel,
        host_device: torch.device,
        checkpoint: str | os.PathLike | Path | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        llm_config: PretrainedConfig | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
    ):
        super().__init__(
            sim_model=sim_model,
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            llm_config=llm_config,
            sequence_length=sequence_length,
            context_length=context_length,
            host_device=host_device,
        )

    @staticmethod
    def get_input_prompt_with_tags(
        user_input_prompt: str = DEFAULT_USER_PROMPT,
        system_context_prompt: str = DEFAULT_PROMPT_CONTEXT,
    ) -> str:
        """
        Get EXAONE4-formatted prompt with role tags
        """
        prompt = f"""{BEGIN_ROLE.format(role=SYSTEM_ROLE)}{system_context_prompt}{END_TURN}{BEGIN_ROLE.format(role=USER_ROLE)}{user_input_prompt}{END_TURN}{BEGIN_ROLE.format(role=ASSISTANT_ROLE)}{THINK_START}

{THINK_END}

"""
        return prompt

    @staticmethod
    def _get_output_names(num_hidden_layers: int):
        """Get output names for ONNX export"""
        output_names = ["logits"]
        for layer in range(num_hidden_layers):
            output_names.append(f"past_key_{layer}_out")
            output_names.append(f"past_value_{layer}_out")
        return output_names

    def _adapt_aimet_encodings(
        self, src_encodings_path: str, dst_encodings_path: str, onnx_model_path: str
    ) -> None:
        """
        Adapt AIMET encodings for EXAONE4 model structure
        """
        with open(src_encodings_path) as f:
            encodings = json.load(f)

        model = onnx.load(onnx_model_path)

        model_input_names = {}
        for node in model.graph.node:
            model_input_names[node.name] = node.input

        uses_lists = Version(encodings["version"]) >= Version("1.0.0")

        if uses_lists:
            # Convert encodings to dictionaries
            encodings["activation_encodings"] = {
                v["name"]: v for v in encodings["activation_encodings"]
            }
            encodings["param_encodings"] = {
                v["name"]: v for v in encodings["param_encodings"]
            }

        # Handle embedding layer encoding
        embed_a_name = "/model/model/embed_tokens/Gather_output_0"
        embed_w_name = "model.model.embed_tokens.weight"
        encodings["activation_encodings"][embed_a_name] = copy.deepcopy(
            encodings["activation_encodings"][embed_w_name]
        )

        for key in encodings["activation_encodings"].keys():
            if "weight" in key:
                encodings["param_encodings"][key] = copy.deepcopy(
                    encodings["activation_encodings"][key]
                )

        if uses_lists:
            encodings["activation_encodings"][embed_a_name]["name"] = embed_a_name

        # Handle layer normalization encodings for EXAONE4
        zero_keys = []
        for layer in range(self.llm_config.num_hidden_layers):
            # EXAONE4 uses post-attention and post-feedforward layernorms
            for norm_type in ["post_attention_layernorm", "post_feedforward_layernorm"]:
                zero_keys += [
                    f"/model/layers.{layer}/{norm_type}/Pow_output_0",
                    f"/model/layers.{layer}/{norm_type}/ReduceMean_output_0",
                    f"/model/layers.{layer}/{norm_type}/Add_output_0",
                    f"/model/layers.{layer}/{norm_type}/Sqrt_output_0",
                    f"/model/layers.{layer}/{norm_type}/Div_output_0",
                    f"/model/layers.{layer}/{norm_type}/Mul_output_0",
                ]
            
            # QK-norm specific encodings
            for norm_type in ["q_norm", "k_norm"]:
                zero_keys += [
                    f"/model/layers.{layer}/self_attn/{norm_type}/Pow_output_0",
                    f"/model/layers.{layer}/self_attn/{norm_type}/ReduceMean_output_0",
                    f"/model/layers.{layer}/self_attn/{norm_type}/Add_output_0",
                    f"/model/layers.{layer}/self_attn/{norm_type}/Sqrt_output_0",
                    f"/model/layers.{layer}/self_attn/{norm_type}/Div_output_0",
                    f"/model/layers.{layer}/self_attn/{norm_type}/Mul_output_0",
                ]

        # Final norm layer
        zero_keys += [
            "/model/norm/Pow_output_0",
            "/model/norm/ReduceMean_output_0",
            "/model/norm/Add_output_0",
            "/model/norm/Sqrt_output_0",
            "/model/norm/Div_output_0",
            "/model/norm/Mul_output_0",
        ]

        # Set zero encodings for normalization operations
        for key in zero_keys:
            if uses_lists:
                zero_entry: Any = {
                    "bw": 16,
                    "dtype": "INT",
                    "enc_type": "PER_TENSOR",
                    "is_sym": False,
                    "name": key,
                    "offset": [0],
                    "scale": [1e-20],
                }
            else:
                zero_entry = [
                    {
                        "bitwidth": 16,
                        "dtype": "int",
                        "is_symmetric": "False",
                        "max": 0.0,
                        "min": 0.0,
                        "offset": 0,
                        "scale": 1e-20,
                    }
                ]
            encodings["activation_encodings"][key] = zero_entry

        propagate_memory_encodings(encodings, model)

        if uses_lists:
            # Convert back to list format
            encodings["activation_encodings"] = list(
                encodings["activation_encodings"].values()
            )
            encodings["param_encodings"] = list(encodings["param_encodings"].values())

        with open(dst_encodings_path, "w") as write_file:
            json.dump(encodings, write_file, indent=4, sort_keys=True)

    @classmethod
    def prepare_ort_genai_assets(
        cls,
        model_name: str,
        llm_config: PretrainedConfig,
        position_processor_cls: type[PositionProcessorBase],
        encodings_path: str | Path,
        context_length: int,
        prompt_sequence_length: int,
        onnx_model_path_from_sub_component_name: dict[str, str],
        num_splits: int,
        qairt_version: str,
        output_dir: str | Path,
    ):
        """Prepare ORT GenAI assets for EXAONE4 model"""
        # This would contain EXAONE4-specific ORT GenAI asset preparation
        # For now, we use a placeholder implementation
        pass
