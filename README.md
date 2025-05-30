# LLMPerfOracle

A virtualized environment for comparative performance analysis of Large Language Model (LLM) serving frameworks.

## Overview

LLMPerfOracle is a discrete-event simulation platform designed to evaluate and compare the performance of various LLM serving frameworks (e.g., vLLM, TensorRT-LLM, SGLang, Dynama) without requiring physical hardware. It provides:

- **Hardware-agnostic simulation**: Test frameworks on virtual hardware profiles without needing GPUs
- **Realistic workload generation**: Model production-like request patterns including bursty arrivals and conversational contexts
- **Framework-specific modeling**: Accurate simulation of batching strategies, KV cache management, and scheduling policies
- **Comprehensive metrics**: Track latency (TTFT, TPOT), throughput, and resource utilization

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
      model_profile_id: "Llama2-7B"
      max_num_seqs: 256
```

## Extending the Simulator

### Adding a New Framework

1. Create a new class inheriting from `AbstractLLMFramework`
2. Implement required methods:
   - `handle_incoming_request()`
   - `processing_loop()`
   - `get_status()`
3. Register the framework in `FRAMEWORK_CLASS_MAP`

### Adding New Workload Patterns

1. Extend `ClientProfile` with new parameters
2. Add distribution configurations in the workload section
3. Implement any special request generation logic

## Model Characteristics Database

The simulator uses a JSON database (`configs/model_params.json`) to store model-specific parameters:

- Model dimensions (hidden size, layers, attention heads)
- Computational requirements (FLOPs per token)
- Memory requirements (KV cache size per token)

Pre-configured models include:
- Llama2-7B, 13B
- Llama3-8B
- Mistral-7B
- GPT-3-175B

## Development

### Running Tests

```bash
# Using pytest
pytest tests/

# Run specific test
pytest tests/test_basic_simulation.py
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

## Future Enhancements

- Additional framework implementations (TensorRT-LLM, SGLang, Dynama)
- Advanced network topology modeling
- Energy consumption modeling
- Multi-node distributed serving simulation
- Integration with real workload traces

## License

[License information to be added]

## Citation

If you use LLMPerfOracle in your research, please cite:

```bibtex
[Citation to be added]
```