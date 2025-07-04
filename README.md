# LLMPerfOracle

A virtualized environment for comparative performance analysis of Large Language Model (LLM) serving frameworks.

## Overview

LLMPerfOracle is a discrete-event simulation platform designed to evaluate and compare the performance of various LLM serving frameworks (e.g., vLLM, TensorRT-LLM, SGLang, Dynama) without requiring physical hardware. It provides:

- **Hardware-agnostic simulation**: Test frameworks on virtual hardware profiles without needing GPUs
- **Realistic workload generation**: Model production-like request patterns including bursty arrivals and conversational contexts
- **Framework-specific modeling**: Accurate simulation of batching strategies, KV cache management, and scheduling policies
- **Comprehensive metrics**: Track latency (TTFT, TPOT), throughput, and resource utilization
- **Performance abstractions**: Configurable Level of Detail (LoD) for faster simulations
- **Advanced features**: Chunked prefill, prefix caching, configuration validation

## Key Features

### Performance Abstractions (Level of Detail)

LLMPerfOracle implements configurable Level of Detail (LoD) from Document 9 to enable faster simulations:

```yaml
simulation:
  lod: "medium"  # Options: "high" (detailed) or "medium" (faster)
```

- **High LoD**: Detailed layer-by-layer simulation for accuracy
- **Medium LoD**: Aggregated operations for 5-20x faster simulations
- Maintains accuracy while reducing simulation time
- Particularly effective with parallelism strategies (TP, PP)

### Chunked Prefill

Handles large prompts that exceed maximum batch size:

```yaml
frameworks_to_test:
  - name: "vllm_chunked"
    type: "VLLM"
    config:
      enable_chunked_prefill: true
      prefill_chunk_size: 4096  # Process large prompts in chunks
```

- Automatically splits large prompts into manageable chunks
- Enables processing of prompts with 10,000+ tokens
- Dynamic batch size calculation based on hardware capabilities
- Essential for heavy workloads with large context windows

### Configuration Validation

Built-in validation system ensures configurations are correct before running:

```python
from llmperforacle.utils.config_validator import ExperimentConfigValidator

is_valid, errors = ExperimentConfigValidator.validate(config)
```

- Validates hardware configurations
- Checks model compatibility
- Ensures network topology is complete
- Prevents common configuration mistakes

## Installation

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd LLMPerfOracle

# Install with Poetry
poetry install
```

### Using pip

```bash
# Clone the repository
git clone <repository-url>
cd LLMPerfOracle

# Install in development mode
pip install -e .
```

## Quick Start

1. **Generate an example configuration:**

```bash
llmperforacle generate-config -o my_config.yaml
```

2. **Run a simulation:**

```bash
llmperforacle run my_config.yaml
```

3. **View results:**

Results are saved to the paths specified in your configuration:
- Summary report: `experiments/results/summary.json`
- Detailed metrics: `experiments/results/requests.csv`

## Architecture

The simulator consists of six main modules:

### 1. Core Simulation Engine
- Built on SimPy for discrete-event simulation
- Manages simulation time and event scheduling

### 2. Parameterized Virtual Hardware Layer
- Models compute devices (GPUs/CPUs) with configurable TFLOPS, memory, and bandwidth
- Simulates network links with latency and bandwidth constraints
- Uses parametric cost models based on Roofline analysis

### 3. Configurable Workload Generator
- Generates request streams based on statistical distributions
- Supports multiple client profiles with different behaviors
- Models conversational patterns and streaming responses

### 4. Pluggable LLM Framework Module
- Abstract interface for implementing different serving frameworks
- Currently implemented: vLLM with PagedAttention and continuous batching
- Extensible design for adding new frameworks

### 5. Metrics Collection and Reporting
- Tracks per-request metrics: TTFT, TPOT, end-to-end latency
- Aggregates system metrics: throughput, GPU utilization
- Exports results in JSON and CSV formats

### 6. Experiment Configuration and Orchestration
- YAML/JSON-based configuration
- Manages simulation setup and execution
- Coordinates all modules

## Configuration

Configuration files define all aspects of the simulation:

```yaml
simulation:
  max_simulation_time: 300  # seconds
  random_seed: 42

hardware_profile:
  compute_devices:
    - device_id: "gpu0"
      device_type: "GPU"
      peak_tflops: {"fp16": 312}
      memory_capacity_bytes: 80_000_000_000
      memory_gbps: 2000

workload:
  client_profiles:
    - profile_name: "interactive_chat"
      inter_arrival_time_dist_config:
        type: "Exponential"
        rate: 10.0  # requests per second

frameworks_to_test:
  - name: "vllm_instance"
    type: "VLLM"
    config:
      model_profile_id: "Llama3-8B"  # Or choose from 15+ available models
      max_num_seqs: 256
```

## Multi-GPU Parallelism Support

LLMPerfOracle now supports simulation of multi-GPU parallelism strategies:

### Tensor Parallelism (TP)
Splits model layers across GPUs, with each GPU computing a portion of each layer:

```yaml
frameworks_to_test:
  - name: "vllm_tp4"
    type: "ParallelVLLM"
    config:
      model_profile_id: "Llama2-13B"
      parallelism:
        strategy: "TP"
        tp_degree: 4
        gpu_ids: ["gpu0", "gpu1", "gpu2", "gpu3"]
```

### Pipeline Parallelism (PP)
Divides model layers into stages, with each stage assigned to different GPUs:

```yaml
frameworks_to_test:
  - name: "vllm_pp2"
    type: "ParallelVLLM"
    config:
      model_profile_id: "GPT-3-175B"
      parallelism:
        strategy: "PP"
        pp_stages: 2
        num_microbatches_per_request: 4
        gpu_ids: ["gpu0", "gpu1"]
```

### Combined TP+PP
Combines both strategies for maximum parallelism:

```yaml
frameworks_to_test:
  - name: "vllm_tp2_pp2"
    type: "ParallelVLLM"
    config:
      model_profile_id: "LargeModel-100B"
      parallelism:
        strategy: "TP_PP"
        tp_degree: 2
        pp_stages: 2
        gpu_ids: ["gpu0", "gpu1", "gpu2", "gpu3"]
```

### Data Parallelism (DP)
Replicates the model across multiple instances for higher throughput:

```yaml
# Define multiple framework instances
frameworks_to_test:
  - name: "replica_1"
    type: "VLLM"
    is_target_for_workload: true
    config:
      model_profile_id: "Llama2-7B"
      gpu_id: "gpu0"
      
  - name: "replica_2"
    type: "VLLM"
    is_target_for_workload: true
    config:
      model_profile_id: "Llama2-7B"
      gpu_id: "gpu1"

# Configure load balancing
workload:
  load_balancing_strategy: "least_loaded"  # Options: round_robin, random, least_loaded, weighted_random, session_affinity
```

### Network Topology for Multi-GPU
Define inter-GPU network links to simulate communication overhead:

```yaml
hardware_profile:
  network_links:
    # NVLink connections within a node
    - link_id: "gpu0_to_gpu1"
      source_id: "gpu0"
      dest_id: "gpu1"
      bandwidth_bps: 600_000_000_000  # 600 GB/s
      latency_s: 0.000001
      bidirectional: true
    
    # InfiniBand connections across nodes
    - link_id: "gpu0_to_gpu4"
      source_id: "gpu0"
      dest_id: "gpu4"
      bandwidth_bps: 200_000_000_000  # 200 GB/s
      latency_s: 0.000005
      bidirectional: true
```

### Complete Multi-GPU Example
See `configs/example_parallel_experiment.yaml` for a comprehensive multi-GPU configuration demonstrating all parallelism strategies.

## Extending the Simulator

### Adding a New Framework

1. Create a new class inheriting from `AbstractLLMFramework`
2. Implement required methods:
   - `handle_incoming_request()`
   - `processing_loop()`
   - `get_status()`
3. For parallelism support, inherit from `ParallelVLLMFramework` or implement your own parallelism logic
4. Register the framework in `FRAMEWORK_CLASS_MAP`

### Adding New Workload Patterns

1. Extend `ClientProfile` with new parameters
2. Add distribution configurations in the workload section
3. Implement any special request generation logic

## Model Characteristics Database

The simulator uses a JSON database (`configs/model_params.json`) to store model-specific parameters:

- Model dimensions (hidden size, layers, attention heads)
- Computational requirements (FLOPs per token)
- Memory requirements (KV cache size per token)
- Per-layer statistics for accurate parallelism simulation
- Tensor parallelism sharding information

Pre-configured models include:
- **Llama Family**: Llama2-7B/13B, Llama3-8B/70B
- **Qwen 2.5 Family**: 7B/32B/72B variants
- **Mistral Family**: Mistral-7B-v0.3, Mixtral-8x7B (MoE)
- **Gemma 2**: 9B/27B models
- **GPT-3-175B**

See [Model Library Documentation](docs/model_library.md) for detailed specifications.

## Development

### Running Tests

#### Regression Test Suite (Recommended)

The project includes a comprehensive regression test suite that ensures all functionality works correctly:

```bash
# Run full regression test suite (~5 minutes)
python run_regression_tests.py

# Run quick regression tests only (~2 minutes)
python run_regression_tests.py --quick

# Run verbose tests with detailed output
python run_regression_tests.py --verbose

# Run specific test category
python run_regression_tests.py --specific unit
python run_regression_tests.py --specific integration
```

The regression suite includes:
- Configuration validation
- Memory validation (ensures models don't exceed GPU memory)
- Parallelism strategy testing (TP, PP, DP)
- Performance abstractions (LoD) verification
- Feature testing (prefix caching, chunked prefill)

#### Using pytest directly

```bash
# Set up environment
export PYTHONPATH=/path/to/LLMPerfOracle/src

# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/              # Unit tests
pytest tests/integration/       # Integration tests
pytest tests/regression/        # Regression tests

# Run quick parallelism tests (demonstrates TP, PP, DP benefits)
pytest tests/integration/test_parallel_simulation_quick.py -v

# Run comprehensive parallelism tests (longer, more thorough)
pytest tests/integration/test_parallel_simulation_improved.py -v
```

### Code Style

The project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

## Limitations

- The simulator provides relative performance comparisons, not absolute performance predictions
- Hardware modeling uses simplified cost models rather than cycle-accurate simulation
- Currently only vLLM framework is implemented; others are planned

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Prefix Caching Support

LLMPerfOracle supports advanced prefix caching simulation with two modes:

### Conversational Prefix Caching (Phase 1)
- Reuses KV cache within conversation sessions
- 70%+ hit rates for multi-turn conversations
- 98% TTFT improvement for cached requests

### Cross-Request Prefix Caching (Phase 2)
- Global cache for common prefixes across requests
- Support for system prompts, few-shot examples
- LRU eviction with configurable cache size

### Configuration
```yaml
frameworks_to_test:
  - name: "vllm_with_caching"
    type: "VLLM"
    config:
      enable_prefix_caching: true        # Conversational caching
      enable_cross_request_caching: true # Global prefix cache
      max_prefix_cache_size: 100         # Max cached prefixes
      min_prefix_cache_length: 50        # Min tokens to cache
```

### Metrics
The simulation reports comprehensive caching statistics:
- Conversational and cross-request hit rates
- Prefill computation reduction (30-70%)
- Per-request token savings
- Cache utilization metrics

See `docs/prefix_caching_implementation.md` for comprehensive details.

## Future Enhancements

- Additional framework implementations (TensorRT-LLM, SGLang, Dynama)
- Advanced collective communication algorithms (NCCL-accurate modeling)
- Heterogeneous hardware support (mixed GPU types)
- Dynamic parallelism adaptation
- Energy consumption modeling
- Multi-node distributed serving simulation
- Integration with real workload traces
- Pipeline bubble optimization strategies
- Expert parallelism for MoE models
- Cross-request prefix caching for shared system prompts
- Advanced KV cache eviction policies

## License

[License information to be added]

## Citation

If you use LLMPerfOracle in your research, please cite:

```bibtex
[Citation to be added]
```