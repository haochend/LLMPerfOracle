# LLMPerfOracle Model Library

## Overview

This document describes the pre-configured model parameters available in LLMPerfOracle. The model library includes both classic and state-of-the-art language models with detailed architectural parameters and performance characteristics.

## Available Models

### Llama Family

#### Llama 2 Series
- **Llama2-7B**: 7B parameters, 32 layers, 32 attention heads
- **Llama2-13B**: 13B parameters, 40 layers, 40 attention heads

#### Llama 3 Series (New)
- **Llama3-8B**: 8B parameters, 32 layers, 32 attention heads, 8 KV heads (GQA)
  - Enhanced with Grouped Query Attention for efficiency
  - Larger vocabulary (128K tokens)
- **Llama3-70B**: 70B parameters, 80 layers, 64 attention heads, 8 KV heads (GQA)
  - State-of-the-art performance with efficient KV cache usage

### Qwen 2.5 Family (New)
- **Qwen2.5-7B**: 7.7B parameters, 28 layers, 28 attention heads, 4 KV heads
  - Efficient architecture with aggressive GQA (7:1 ratio)
  - Large vocabulary (152K tokens)
- **Qwen2.5-32B**: 32.5B parameters, 64 layers, 40 attention heads, 8 KV heads
  - Balanced mid-size model with strong performance
- **Qwen2.5-72B**: 72.7B parameters, 80 layers, 64 attention heads, 8 KV heads
  - Flagship model competitive with GPT-4 class models

### Mistral Family
- **Mistral-7B-v0.3**: 7B parameters, 32 layers, 32 attention heads, 8 KV heads
  - Efficient architecture with sliding window attention
- **Mixtral-8x7B** (New): 46.7B parameters (sparse), 32 layers, 8 experts
  - Mixture of Experts architecture
  - Only 2 experts active per token (efficient inference)

### Gemma 2 Family (New)
- **Gemma2-9B**: 9.2B parameters, 42 layers, 16 attention heads, 8 KV heads
  - Optimized for edge deployment
  - Large vocabulary (256K tokens)
- **Gemma2-27B**: 27.2B parameters, 46 layers, 32 attention heads, 16 KV heads
  - Strong performance in mid-size category

### GPT Family
- **GPT-3-175B**: 175B parameters, 96 layers, 96 attention heads
  - Classic large-scale transformer

## Model Features

### Grouped Query Attention (GQA)
Many newer models use GQA to reduce KV cache memory:
- Llama3 models: 4:1 or 8:1 compression
- Qwen models: Up to 7:1 compression
- Gemma 2: 2:1 compression

### Mixture of Experts (MoE)
MoE models provide larger capacity with efficient inference:
- **Mixtral-8x7B**: 8 experts, 2 active per token

### Vocabulary Sizes
- Standard (32K): Llama2, Mistral
- Large (128K): Llama3
- Extra Large (152K): Qwen 2.5
- Massive (256K): Gemma 2

## Usage Example

```yaml
frameworks_to_test:
  - name: "vllm_llama3"
    type: "VLLM"
    config:
      model_profile_id: "Llama3-70B"  # Use new Llama3-70B model
      gpu_id: "gpu0"
      tensor_parallel_degree: 4  # Recommended for 70B model
```

## Performance Characteristics

### Compute Requirements (FLOPs per token)
- Small models (7-9B): 14-18 TFLOPs
- Medium models (32-70B): 65-145 TFLOPs  
- Large models (100B+): 200+ TFLOPs

### Memory Bandwidth Requirements
- Prefill: 2x parameter size per token
- Decode: 4x parameter size per token (includes KV cache)

### KV Cache Optimization
Models with aggressive GQA ratios significantly reduce memory usage:
- Standard (1:1): Full KV cache per head
- Moderate GQA (4:1): 75% reduction
- Aggressive GQA (8:1): 87.5% reduction

## Recommendations

### Model Selection Guidelines
1. **For latency-sensitive applications**: Llama3-8B, Qwen2.5-7B, Mistral-7B
2. **For quality-focused applications**: Llama3-70B, Qwen2.5-72B
3. **For throughput optimization**: Mixtral-8x7B (MoE efficiency)
4. **For memory-constrained environments**: Models with high GQA ratios
5. **For balanced performance**: Qwen2.5-32B, Gemma2-27B

### Hardware Matching
- **Single GPU (80GB)**: All models up to 32B, some 70B with quantization
- **Multi-GPU required**: 70B+ models at full precision
- **Tensor Parallelism recommended**: 
  - 2 GPUs: 32B models
  - 4 GPUs: 70B models
  - 8+ GPUs: 100B+ models

## Adding New Models

To add a new model to the library, edit `configs/model_params.json` with:
1. Basic architecture parameters
2. Compute and memory statistics
3. Layer-wise characteristics
4. Parallelism support flags

See existing models for examples of the required format.