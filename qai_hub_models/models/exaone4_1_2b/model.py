# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path

import torch

from qai_hub_models.models._shared.exaone4.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    Exaone4Base,
    Exaone4Base_AIMETOnnx,
)
from qai_hub_models.models._shared.llm.model import determine_precision_from_checkpoint
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# EXAONE4 1.2B model configuration based on config.json and README
NUM_LAYERS = 30
NUM_SPLITS = 3
NUM_LAYERS_PER_SPLIT = 10  # 30 layers / 3 splits
HIDDEN_SIZE = 2048
NUM_KEY_VALUE_HEADS = 8
NUM_ATTN_HEADS = 32
HEAD_DIM = 64
VOCAB_SIZE = 102400
INTERMEDIATE_SIZE = 4096  # From config.json
MAX_POSITION_EMBEDDINGS = 65536  # From config.json

# Hugging Face repo name and URL
HF_REPO_NAME = "LGAI-EXAONE/EXAONE-4.0-1.2B"
HF_REPO_URL = f"https://huggingface.co/{HF_REPO_NAME}"

# Minimum memory (RAM+swap) recommended for export (in GB)
MIN_MEMORY_RECOMMENDED = 25

# Default precision settings
DEFAULT_PRECISION = Precision.w4
SUPPORTED_PRECISIONS = [Precision.w4, Precision.w4a16]
DEFAULT_CHECKPOINT = {
    Precision.w4: "exaone4_ckpt_w4",
    Precision.w4a16: "exaone4_ckpt_w4a16",
}


class Exaone4_1_2B(Exaone4Base):
    """EXAONE4 1.2B model implementation"""
    
    min_memory_recommended = MIN_MEMORY_RECOMMENDED

    def __init__(
        self,
        checkpoint: str | os.PathLike | Path = HF_REPO_NAME,
        *args,
        **kwargs,
    ):
        super().__init__(
            checkpoint=checkpoint,  # type: ignore[misc]
            *args,
            **kwargs,
        )

    def _verify_ckpt(self):
        """Verify that the checkpoint is compatible with EXAONE4 1.2B"""
        super()._verify_ckpt()
        if not (
            self.llm_config.num_hidden_layers == NUM_LAYERS
            and self.llm_config.hidden_size == HIDDEN_SIZE
            and self.llm_config.num_attention_heads == NUM_ATTN_HEADS
            and self.llm_config.num_key_value_heads == NUM_KEY_VALUE_HEADS
        ):
            raise ValueError("Model config is not compatible with EXAONE4 1.2B implementation.")

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | os.PathLike | Path = HF_REPO_NAME,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        host_device: torch.device | None = None,
        load_pretrained: bool = True,
        _skip_optimizations: list[str] | None = None,
    ) -> Exaone4_1_2B:
        """
        Load a pre-trained EXAONE4 1.2B model from LG AI Research via HuggingFace.

        checkpoint:
            Local path or Hugging Face name of floating point checkpoint.
        sequence_length:
            Instantiate with this token sequence length input. A longer
            sequence length means the model is capable of processing more
            tokens at once. This can only be set to greater than one to process
            prompts, since responses are auto-regressive in nature and require
            this to be 1.
        context_length:
            Total context length of model. Longer context length means the
            model is more capable of making longer connections in the input
            prompt. However, it also hurts runtime performance (both time-to-
            first-token and tokens-per-second), so this is a tradeoff that may
            depend on the use case. EXAONE4 supports up to 65536 tokens.
        host_device:
            Device on which to load the model.
        load_pretrained:
            Whether to load pretrained weights.
        _skip_optimizations:
            List of optimizations to skip. Options include:
            - "sha_attention": Skip Split-Head Attention optimization
            - "rank4_rms_norm": Skip rank-4 RMS norm optimization
            - "qk_norm": Skip QK normalization optimization
        """
        return cls(
            checkpoint=checkpoint,
            sequence_length=sequence_length,
            context_length=context_length,
            host_device=host_device,
            load_pretrained=load_pretrained,
            _skip_optimizations=_skip_optimizations,
        )

    @staticmethod
    def get_output_names():
        """Get output names for ONNX export"""
        return Exaone4Base._get_output_names(NUM_LAYERS)

    @staticmethod
    def get_input_spec(
        llm_config: dict,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
    ) -> InputSpec:
        """Get input specification for the model"""
        return Exaone4Base._get_input_spec(
            num_hidden_layers=llm_config["num_hidden_layers"],
            sequence_length=sequence_length,
            context_length=context_length,
            hidden_size=llm_config["hidden_size"],
            num_key_value_heads=llm_config["num_key_value_heads"],
            num_attention_heads=llm_config["num_attention_heads"],
        )


class Exaone4_1_2B_AIMETOnnx(Exaone4Base_AIMETOnnx):
    """EXAONE4 1.2B AIMET ONNX model implementation"""

    def __init__(self, checkpoint: str | os.PathLike | Path | None, *args, **kwargs):
        super().__init__(
            checkpoint=checkpoint,  # type: ignore[misc]
            *args,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | os.PathLike | Path | None = "DEFAULT",
        host_device: torch.device | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        precision: Precision = DEFAULT_PRECISION,
        fp_model: torch.nn.Module | None = None,
        _skip_quantsim_creation: bool = False,
    ) -> Exaone4_1_2B_AIMETOnnx:
        """
        Load EXAONE4 1.2B model and create AIMET-ONNX QuantSim.
        Optionally load ONNX model and AIMET encodings from a checkpoint.

        Args:
            checkpoint: Path to previously calibrated AIMET encodings and ONNX
                models. Note that encodings are sensitive to AIMET ONNX versions.
                If passing None, initializes without encodings.
            host_device: Device on which to load the model.
            sequence_length: Input sequence length for the model.
            context_length: Maximum context length the model can handle.
            precision: Quantization precision to use.
            fp_model: Floating point model instance for quantization.
            _skip_quantsim_creation: Skip QuantSim creation (for export only).
        """
        if isinstance(checkpoint, str) and checkpoint.startswith("DEFAULT"):
            precision = determine_precision_from_checkpoint(checkpoint) or precision
            if precision not in SUPPORTED_PRECISIONS:
                available_precisions = [str(p) for p in SUPPORTED_PRECISIONS]
                raise ValueError(
                    f"This model is not supported for {str(precision)} precision. "
                    f"Models are available in following precisions: {','.join(available_precisions)}."
                )
            if precision not in DEFAULT_CHECKPOINT:
                available_checkpoints = [str(p) for p in DEFAULT_CHECKPOINT]
                raise ValueError(
                    f"No checkpoint is available for this model in {str(precision)} precision. If you would "
                    f"like to continue with this precision, please generate a local quantized checkpoint. "
                    f"Checkpoints are available in the following precisions: {','.join(available_checkpoints)}."
                )
            precision_checkpoint = DEFAULT_CHECKPOINT[precision]
            checkpoint = os.path.join(
                CachedWebModelAsset.from_asset_store(
                    MODEL_ID, MODEL_ASSET_VERSION, precision_checkpoint + ".zip"
                ).fetch(extract=True),
                precision_checkpoint,
            )
            # Generate necessary ONNX models
            if fp_model is not None:
                cls.create_onnx_models(
                    checkpoint=checkpoint,
                    fp_model=fp_model,
                    context_length=context_length,
                    export_sequence_lengths=[sequence_length],
                    host_device=host_device,
                )

                cls.save_tokenizer_and_config(checkpoint=checkpoint, fp_model=fp_model)
        
        return super().from_pretrained(
            checkpoint=checkpoint,
            host_device=host_device,
            sequence_length=sequence_length,
            context_length=context_length,
            precision=precision,
            fp_model=fp_model,
            _skip_quantsim_creation=_skip_quantsim_creation,
        )

    @staticmethod
    def get_output_names():
        """Get output names for ONNX export"""
        return Exaone4Base._get_output_names(NUM_LAYERS)

    @staticmethod
    def get_input_spec(
        llm_config: dict,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
    ) -> InputSpec:
        """Get input specification for the quantized model"""
        return Exaone4Base._get_input_spec(
            num_hidden_layers=llm_config["num_hidden_layers"],
            sequence_length=sequence_length,
            context_length=context_length,
            hidden_size=llm_config["hidden_size"],
            num_key_value_heads=llm_config["num_key_value_heads"],
            num_attention_heads=llm_config["num_attention_heads"],
        )
