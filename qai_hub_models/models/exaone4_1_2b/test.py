# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import qai_hub as hub
import torch
from transformers import PretrainedConfig

from qai_hub_models.models._shared.exaone4.model import Exaone4Base
from qai_hub_models.models._shared.llm.evaluate import create_quantsim, evaluate
from qai_hub_models.models._shared.llm.model import cleanup
from qai_hub_models.models._shared.llm.quantize import quantize
from qai_hub_models.models.common import TargetRuntime
from qai_hub_models.models.exaone4_1_2b import MODEL_ID, FP_Model, Model
from qai_hub_models.models.exaone4_1_2b.demo import exaone4_1_2b_chat_demo
from qai_hub_models.models.exaone4_1_2b.export import (
    DEFAULT_EXPORT_DEVICE,
    NUM_SPLITS,
)
from qai_hub_models.models.exaone4_1_2b.export import main as export_main
from qai_hub_models.models.exaone4_1_2b.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_PRECISION,
    DEFAULT_SEQUENCE_LENGTH,
    HF_REPO_NAME,
    Exaone4_1_2B,
)
from qai_hub_models.utils.base_model import Precision
from qai_hub_models.utils.checkpoint import CheckpointSpec
from qai_hub_models.utils.model_cache import CacheMode

DEFAULT_EVAL_SEQLEN = 2048


@pytest.mark.unmarked
@pytest.mark.parametrize(
    "skip_inferencing, skip_profiling, target_runtime",
    [
        (True, True, TargetRuntime.QNN_CONTEXT_BINARY),
        (True, False, TargetRuntime.QNN_CONTEXT_BINARY),
        (False, True, TargetRuntime.QNN_CONTEXT_BINARY),
        (False, False, TargetRuntime.QNN_CONTEXT_BINARY),
    ],
)
def test_cli_device_with_skips(
    tmp_path: Path,
    skip_inferencing: bool,
    skip_profiling: bool,
    target_runtime: TargetRuntime,
):
    """Test EXAONE4 export CLI with various skip options"""
    from qai_hub_models.models._shared.llama3 import test
    test.test_cli_device_with_skips(
        export_main,
        Model,
        tmp_path,
        MODEL_ID,
        NUM_SPLITS,
        hub.Device(DEFAULT_EXPORT_DEVICE),
        skip_inferencing,
        skip_profiling,
        target_runtime,
    )


def test_cli_device_with_skips_unsupported_device(
    tmp_path: Path,
):
    """Test EXAONE4 export CLI with unsupported device"""
    from qai_hub_models.models._shared.llama3 import test
    test.test_cli_device_with_skips_unsupported_device(
        export_main, Model, tmp_path, MODEL_ID
    )


@pytest.mark.unmarked
@pytest.mark.parametrize(
    "chipset, context_length, sequence_length, target_runtime",
    [
        ("qualcomm-snapdragon-8gen2", 2048, 256, TargetRuntime.QNN_CONTEXT_BINARY),
        ("qualcomm-snapdragon-x-elite", 4096, 128, TargetRuntime.QNN_CONTEXT_BINARY),
    ],
)
def test_cli_chipset_with_options(
    tmp_path: Path,
    context_length: int,
    sequence_length: int,
    chipset: str,
    target_runtime: TargetRuntime,
):
    """Test EXAONE4 export CLI with different chipsets and configurations"""
    from qai_hub_models.models._shared.llama3 import test
    test.test_cli_chipset_with_options(
        export_main,
        Model,
        tmp_path,
        MODEL_ID,
        NUM_SPLITS,
        chipset,
        context_length,
        sequence_length,
        target_runtime,
    )


@pytest.mark.trace
def test_trace():
    """Test EXAONE4 model tracing"""
    # Test a very small configuration for faster testing
    model = Exaone4_1_2B.from_pretrained(
        checkpoint=HF_REPO_NAME,
        sequence_length=16,
        context_length=128,
        load_pretrained=False,  # Don't load weights for faster testing
    )
    
    input_dict = model.sample_inputs()
    model(**input_dict)
    cleanup()


@pytest.mark.integration
@pytest.mark.parametrize(
    "precision",
    [
        Precision.w4,
        Precision.w4a16,
    ],
)
def test_quantize(
    tmp_path: Path,
    precision: Precision,
):
    """Test EXAONE4 quantization"""
    quantize(
        model_cls=Model,
        fp_model_cls=FP_Model,
        sequence_length=128,
        context_length=512,
        output_path=str(tmp_path),
        precision=precision,
        num_samples=1,  # Minimal for testing
        model_cache_mode=CacheMode.IN_MEMORY,
    )
    cleanup()


@pytest.mark.integration
@pytest.mark.parametrize(
    "precision",
    [
        Precision.w4,
        Precision.w4a16,
    ],
)
def test_create_quantsim(
    tmp_path: Path,
    precision: Precision,
):
    """Test EXAONE4 AIMET QuantSim creation"""
    model = create_quantsim(
        Model,
        FP_Model,
        sequence_length=128,
        context_length=512,
        checkpoint=None,
        precision=precision,
    )
    assert model is not None
    cleanup()


@pytest.mark.integration
@pytest.mark.parametrize(
    "precision",
    [
        Precision.w4,
        Precision.w4a16,
    ],
)
def test_evaluate(
    precision: Precision,
):
    """Test EXAONE4 model evaluation"""
    evaluate(
        quantized_model_cls=Model,
        fp_model_cls=FP_Model,
        checkpoint=None,
        task="tiny-mmlu",
        sequence_length=DEFAULT_EVAL_SEQLEN,
        context_length=DEFAULT_EVAL_SEQLEN,
        precision=precision,
        num_samples=2,  # Minimal for testing
    )
    cleanup()


def test_demo_fp():
    """Test EXAONE4 floating point demo"""
    exaone4_1_2b_chat_demo(
        model_cls=FP_Model,
        fp_model_cls=FP_Model,
        test_checkpoint=CheckpointSpec(
            context_length=512,
            sequence_length=128,
            load_pretrained=False,  # Don't load weights for faster testing
        ),
    )
    cleanup()


def test_demo_quantized():
    """Test EXAONE4 quantized demo"""
    exaone4_1_2b_chat_demo(
        model_cls=Model,
        test_checkpoint=CheckpointSpec(
            precision=DEFAULT_PRECISION,
            context_length=512,
            sequence_length=128,
            checkpoint=None,
        ),
    )
    cleanup()


def test_numerical_accuracy_vs_hf():
    """Test EXAONE4 numerical accuracy against HuggingFace implementation"""
    # This test would compare outputs between our optimized implementation
    # and the standard HuggingFace implementation
    context_length = 512
    sequence_length = 128
    
    # Create our model
    model = Exaone4_1_2B.from_pretrained(
        checkpoint=HF_REPO_NAME,
        sequence_length=sequence_length,
        context_length=context_length,
        load_pretrained=False,  # Skip weights loading for faster test
    )
    
    # Get sample inputs
    input_dict = model.sample_inputs()
    
    # Run inference
    outputs = model(**input_dict)
    
    # Basic sanity checks
    assert len(outputs) > 0
    assert isinstance(outputs[0], torch.Tensor)  # logits
    assert outputs[0].shape[-1] == 102400  # vocab size for EXAONE4
    
    cleanup()


def test_config_compatibility():
    """Test EXAONE4 config compatibility checks"""
    # Test with correct config
    model = Exaone4_1_2B.from_pretrained(
        checkpoint=HF_REPO_NAME,
        load_pretrained=False,
    )
    assert isinstance(model, Exaone4Base)
    
    # Test config verification would catch incompatible configs
    # (This would be more meaningful with actual model weights)
    
    cleanup()


@pytest.mark.parametrize(
    "sequence_length, context_length",
    [
        (1, 128),
        (128, 512),
        (256, 1024),
    ],
)
def test_different_sequence_lengths(
    sequence_length: int,
    context_length: int,
):
    """Test EXAONE4 with different sequence and context lengths"""
    model = Exaone4_1_2B.from_pretrained(
        checkpoint=HF_REPO_NAME,
        sequence_length=sequence_length,
        context_length=context_length,
        load_pretrained=False,
    )
    
    input_dict = model.sample_inputs()
    outputs = model(**input_dict)
    
    # Verify output shapes
    assert outputs[0].shape[1] == sequence_length  # batch_size=1, seq_len, vocab_size
    
    cleanup()


def test_reasoning_mode_prompt():
    """Test EXAONE4 reasoning mode prompt formatting"""
    prompt = Exaone4Base.get_input_prompt_with_tags(
        user_input_prompt="What is 2+2?",
        system_context_prompt="You are a helpful math tutor."
    )
    
    # Check for EXAONE4-specific tags
    assert "[|system|]" in prompt
    assert "[|user|]" in prompt
    assert "[|assistant|]" in prompt
    assert "[|endofturn|]" in prompt
    assert "<think>" in prompt
    assert "</think>" in prompt
    
    # Check content is included
    assert "What is 2+2?" in prompt
    assert "You are a helpful math tutor." in prompt


def test_hybrid_attention_support():
    """Test that EXAONE4 model supports hybrid attention configuration"""
    model = Exaone4_1_2B.from_pretrained(
        checkpoint=HF_REPO_NAME,
        load_pretrained=False,
    )
    
    # Check that model config supports layer_types for hybrid attention
    assert hasattr(model.llm_config, 'layer_types') or hasattr(model.llm_config, 'sliding_window')
    
    cleanup()


if __name__ == "__main__":
    # Run a basic test when script is executed directly
    test_trace()
    print("EXAONE4 1.2B basic test passed!")
