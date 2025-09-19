# EXAONE-4.0-1.2B

## Use case
EXAONE 4.0 is a state-of-the-art large language model developed by LG AI Research that integrates both non-reasoning and reasoning modes to deliver exceptional performance across a wide range of language understanding and generation tasks.

## Description

EXAONE 4.0 represents a significant advancement in language model architecture, featuring:

### Key Features
- **Hybrid Reasoning**: Combines non-reasoning mode for fast inference and reasoning mode with `<think>` blocks for complex problem-solving
- **Multilingual Support**: Native support for English, Korean, and Spanish
- **Tool Use Capabilities**: Built-in support for function calling and tool execution
- **Advanced Architecture**: Features QK-Reorder-Norm and hybrid attention patterns
- **Long Context**: Supports up to 65,536 tokens context length

### Technical Specifications
- **Model Size**: 1.28B parameters
- **Architecture**: Transformer decoder with hybrid attention
- **Layers**: 30 transformer layers
- **Hidden Size**: 2048
- **Attention Heads**: 32 (with 8 key-value heads for GQA)
- **Head Dimension**: 64
- **Vocabulary**: 102,400 tokens
- **Context Length**: 65,536 tokens
- **Quantization**: w4 + w8 (selective layers) + fp16 activations

### Architecture Innovations

1. **Hybrid Attention**: Alternates between full attention and sliding window attention layers in a 3:1 ratio for optimal performance
2. **QK-Reorder-Norm**: Applies LayerNorm directly to attention and MLP outputs with RMS normalization after Q and K projections
3. **Global NoPE**: Uses global attention without RoPE for better global context understanding
4. **Optimized Quantization**: Strategic quantization with 4-bit weights and 8-bit activations for efficient on-device deployment

## Installation

```bash
pip install qai-hub-models
```

For development or to run quantization:
```bash
pip install "qai-hub-models[dev]"
```

## Usage

### Basic Usage

```python
from qai_hub_models.models.exaone4_1_2b import Model

# Load quantized model
model = Model.from_pretrained()

# Load floating point model  
from qai_hub_models.models.exaone4_1_2b import FP_Model
fp_model = FP_Model.from_pretrained()
```

### Chat Interface

```python
from qai_hub_models.models.exaone4_1_2b.demo import exaone4_1_2b_chat_demo

# Run interactive chat demo
exaone4_1_2b_chat_demo()
```

### Reasoning Mode

EXAONE 4.0 supports reasoning mode with explicit thinking steps:

```python
# Enable reasoning mode in tokenizer
messages = [
    {"role": "user", "content": "Which one is bigger, 3.12 vs 3.9?"}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=True,  # Enable reasoning mode
)
```

### Tool Use

EXAONE 4.0 has built-in tool calling capabilities:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "parameters": {
            "type": "object",
            "required": ["expression"],
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            }
        }
    }
}]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    tools=tools,
)
```

## Performance

The model is optimized for deployment on Qualcomm devices with excellent performance characteristics:

| Device | Precision | TTFT (ms) | Tokens/sec |
|--------|-----------|-----------|------------|
| Snapdragon 8 Elite | w4a16 | 43.5-1390 | 48.2 |
| Snapdragon 8 Elite | w4 | 95-3050 | 30.5 |
| SA8295P | w4 | 265-8500 | 9.8 |

*TTFT = Time To First Token (range depends on prompt length)*

## Model Export

```bash
python -m qai_hub_models.models.exaone4_1_2b.export --precision w4a16
```

## Model Quantization

```bash
python -m qai_hub_models.models.exaone4_1_2b.quantize --precision w4a16
```

## Model Evaluation

```bash
python -m qai_hub_models.models.exaone4_1_2b.evaluate --task mmlu
```

## Supported Precisions

- **w4**: 4-bit weights with float16 activations
- **w4a16**: 4-bit weights with 16-bit symmetric activations (recommended)

## Use Cases

- **Dialogue Systems**: Conversational AI with reasoning capabilities
- **Content Generation**: Creative writing, summarization, translation
- **Code Generation**: Programming assistance and code explanation  
- **Mathematical Problem Solving**: Step-by-step reasoning for math problems
- **Tool-assisted Tasks**: Complex workflows requiring function calls
- **Multilingual Applications**: Cross-language communication and translation
- **Educational Assistance**: Tutoring with explanatory reasoning

## License

The model is subject to the EXAONE AI Model License Agreement. Please see the [license file](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B/blob/main/LICENSE) for details.

## Citation

```bibtex
@article{exaone-4.0,
  title={EXAONE 4.0: Unified Large Language Models Integrating Non-reasoning and Reasoning Modes},
  author={{LG AI Research}},
  journal={arXiv preprint arXiv:2507.11407},
  year={2025}
}
```

## Model Card

For more details about the model's capabilities, limitations, and training data, please refer to the [model card](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B).
