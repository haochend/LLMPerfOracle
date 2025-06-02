# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLMPerfOracle is a discrete-event simulation platform for comparing LLM serving frameworks performance without physical hardware. It uses SimPy for event simulation and implements detailed models of serving frameworks like vLLM.

## High-Level Architecture

The system consists of six interconnected modules:

1. **Core Simulation Engine** (`src/llmperforacle/core/`)
   - Wraps SimPy environment
   - Manages simulation time and event scheduling
   - Entry point: `SimulationEnvironment` class

2. **Virtual Hardware Layer** (`src/llmperforacle/hardware/`)
   - Models compute devices and network links
   - Uses parametric cost models for operation timing
   - Key classes: `VirtualHardwarePlatform`, `VirtualComputeDevice`

3. **Workload Generator** (`src/llmperforacle/workload/`)
   - Generates realistic request streams
   - Supports statistical distributions and conversational patterns
   - Key class: `WorkloadGenerator`

4. **Framework Modules** (`src/llmperforacle/frameworks/`)
   - Abstract base: `AbstractLLMFramework`
   - Implementations: `VLLMFramework` (PagedAttention, continuous batching)
   - Each framework manages its own scheduling and resource allocation

5. **Metrics Collection** (`src/llmperforacle/metrics/`)
   - Tracks per-request and system-wide metrics
   - Calculates TTFT, TPOT, throughput
   - Key class: `MetricsCollector`

6. **Orchestration** (`src/llmperforacle/orchestration/`)
   - Loads configuration and coordinates all modules
   - Entry point: `ExperimentOrchestrator`

## Key Design Patterns

### SimPy Process Pattern
Most components use SimPy generator functions that yield events:
```python
def processing_loop(self):
    while True:
        # Do work
        yield self.simpy_env.timeout(duration)
        # More work
        yield resource.request()
```

### Framework Adapter Pattern
New serving frameworks implement `AbstractLLMFramework` interface:
- `handle_incoming_request()`: Accept requests
- `processing_loop()`: Main scheduling logic
- `get_status()`: Report internal state

### Cost Model Pattern
Virtual hardware estimates operation time based on:
- FLOPs for compute-bound operations (prefill)
- Memory bandwidth for memory-bound operations (decode)
- Roofline model to determine bottleneck

## Common Development Tasks

### Adding a New LLM Framework

1. Create new file in `src/llmperforacle/frameworks/`
2. Inherit from `AbstractLLMFramework`
3. Implement required methods
4. Add to `FRAMEWORK_CLASS_MAP` in `orchestration/experiment_orchestrator.py`
5. For parallelism support, inherit from `ParallelVLLMFramework` or implement custom parallelism

### Adding New Workload Distributions

1. Extend `DistributionSampler` in `workload/sampler.py` if needed
2. Add new distribution configs to workload YAML
3. Update `ClientProfile` if new parameters needed

### Running Tests

**IMPORTANT: Always activate the virtual environment before running any Python commands:**

```bash
# Set up environment
cd /Users/davidd/LLMPerfOracle
source venv/bin/activate
export PYTHONPATH=/Users/davidd/LLMPerfOracle/src

# Run all tests
pytest tests/

# Run with coverage
pytest --cov=llmperforacle tests/

# Run specific test
pytest tests/test_basic_simulation.py -v

# Run parallelism integration tests
pytest tests/integration/test_parallel_simulation_quick.py -v

# Run comprehensive parallelism tests
pytest tests/integration/test_parallel_simulation_improved.py -v
```

### Running Regression Tests

The project includes a comprehensive regression test suite to ensure system stability after changes:

```bash
# Run full regression test suite
./run_regression_tests.py

# Run quick regression tests only (faster)
./run_regression_tests.py --quick

# Run with verbose output
./run_regression_tests.py --verbose

# Run specific test
./run_regression_tests.py --specific test_memory_validation

# Skip certain test types
./run_regression_tests.py --no-unit      # Skip unit tests
./run_regression_tests.py --no-regression # Skip regression tests
```

The regression suite tests:
- **Configuration Validation**: All example configs are valid
- **Model Database**: All models have correct parameters
- **Memory Validation**: Models fail appropriately when they don't fit
- **Parallelism**: TP, PP, and mixed strategies work correctly
- **Performance**: PP latency is reasonable (not 55x worse)
- **LoD Speedup**: Medium LoD provides >20x event reduction (aggressive threshold)
- **Features**: Prefix caching, chunked prefill work correctly

Run regression tests after:
- Modifying parallelism implementations
- Changing configuration validation
- Updating model parameters
- Modifying framework behavior
- Any significant refactoring

### Debugging Simulations

1. Set log level to DEBUG in CLI: `llmperforacle run config.yaml -l DEBUG`
2. Add logging statements in SimPy processes
3. Use `get_status()` methods to inspect framework state
4. Check metrics collector for detailed request traces

## Important Implementation Details

### Memory Management in vLLM
- Uses SimPy Container for KV cache blocks
- Blocks allocated during admission, released on completion
- PagedAttention simulated with block-based allocation
- Pipeline Parallelism: Each stage only allocates KV cache for its assigned layers

### Multi-GPU Parallelism
- **Tensor Parallelism (TP)**: Shards model layers across GPUs with collective operations
- **Pipeline Parallelism (PP)**: Splits model into stages with microbatching
- **Data Parallelism (DP)**: Multiple model replicas with load balancing
- **Combined TP+PP**: Hierarchical parallelism for large models

### Prefix Caching (KV Cache Reuse)
- Conversational requests can reuse KV cache from previous turns
- Reduces prefill computation for shared prompt prefixes
- Tracks session state in `active_sessions_kv_state`
- Enable with `enable_prefix_caching: true` in framework config

### Request Flow
1. WorkloadGenerator creates Request object
2. Simulates network transfer to framework
3. Framework queues request in `request_arrival_queue`
4. Framework's `processing_loop` admits based on resources
5. Prefill → Decode loop until completion
6. Metrics logged at each stage

### Time Modeling
- All times in simulation seconds (floats)
- Hardware operations return SimPy timeout events
- Actual duration calculated by cost models
- Metrics converted to milliseconds for reporting

## Configuration Schema

Key configuration sections:
- `simulation`: Max time, random seed
- `hardware_profile`: Virtual devices and network
- `workload`: Client profiles and distributions  
- `frameworks_to_test`: Framework instances and configs
- `metrics_config`: Output paths and statistics

## Tips for Development

1. **Always use absolute paths** in hardware operations
2. **Yield SimPy events** for any time-consuming operations
3. **Log at DEBUG level** for detailed operation tracking
4. **Test with small simulations first** (10-30 seconds)
5. **Profile memory usage** for large-scale simulations
6. **Use type hints** for better IDE support
7. **Keep cost models simple** - relative accuracy matters more than absolute

## Common Pitfalls

- Forgetting to yield SimPy events (causes instant completion)
- Wrapping SimPy Process objects in another process() call (causes "not a generator" error)
- Modifying shared state without proper synchronization
- Not releasing resources (causes deadlocks)
- Using relative paths in file operations
- Not handling edge cases in workload generation
- Missing reverse network links in hardware configuration

## Fixing Common Issues

### "Process is not a generator" Error
This occurs when wrapping a SimPy process in another process call:
```python
# Wrong
yield self.simpy_env.process(
    self.virtual_hardware.submit_computation_task(...)  # Already returns a Process
)

# Correct
yield self.virtual_hardware.submit_computation_task(...)
```

### LoD Not Being Applied
If LoD (Level of Detail) setting is not being respected:
- Check that `SimulationEnvironment` sets metadata: `self.env.metadata = {'lod': self.lod}`
- Ensure `ExperimentOrchestrator` doesn't overwrite metadata after initialization
- Frameworks access LoD via: `simpy_env.metadata['lod']`
- Use regression test `TestLoDRegression::test_lod_event_reduction` to verify LoD works

### Network Link Warnings
Add bidirectional links in hardware config:
```yaml
network_links:
  - link_id: "client_to_server"
    source_id: "client_node_0"
    dest_id: "framework_entry_0"
    bidirectional: true  # Creates reverse link automatically
```

### Low Success Rate in Simulations
- Reduce request rate in workload config
- Increase `max_num_seqs` in framework config
- Increase GPU memory in hardware profile
- Check KV cache block size vs. token lengths

### Testing Parallelism Benefits
To see benefits from parallelism, ensure workloads are appropriately sized:
- **TP benefits**: Use compute-heavy workloads (large prompts, 1000+ tokens)
- **PP benefits**: Use memory-constrained scenarios (long sequences, reduced GPU memory)
- **DP benefits**: Use high request rates that saturate single instances (25+ req/s)

## Performance Optimization

- Batch operations when possible
- Use numpy for statistical operations
- Minimize logging in hot paths
- Pre-calculate model characteristics
- Use efficient data structures for queues

## Project Structure

```
LLMPerfOracle/
├── src/llmperforacle/
│   ├── core/            # Simulation engine
│   ├── hardware/        # Virtual hardware models
│   ├── workload/        # Request generation
│   ├── frameworks/      # LLM framework simulations
│   ├── metrics/         # Performance tracking
│   ├── orchestration/   # Experiment management
│   └── cli.py          # Command-line interface
├── configs/             # Example configurations
│   ├── model_params.json
│   └── example_experiment.yaml
├── tests/              # Test suite
├── experiments/        # Results directory
├── docs/               # Documentation
│   └── design_documents/  # Original design docs
├── venv/              # Virtual environment
├── pyproject.toml     # Poetry configuration
├── setup.py           # Setup script
├── requirements.txt   # Dependencies
├── README.md          # User documentation
└── CLAUDE.md         # This file
```

## CLI Usage

**IMPORTANT: Always activate the virtual environment before running any Python commands:**

```bash
# Activate virtual environment first
cd /Users/davidd/LLMPerfOracle
source venv/bin/activate
export PYTHONPATH=/Users/davidd/LLMPerfOracle/src

# Generate example configuration
python -m llmperforacle.cli generate-config -o my_config.yaml

# Validate configuration before running
python -m llmperforacle.cli validate my_config.yaml
python -m llmperforacle.cli validate my_config.yaml -m  # Also validate model DB

# Run simulation
python -m llmperforacle.cli run my_config.yaml -l INFO

# With custom log level
python -m llmperforacle.cli run my_config.yaml -l DEBUG
```

## Configuration Validation

The system includes comprehensive configuration validation to catch issues before running simulations:

- **Automatic validation** when loading configurations
- **Memory checks** to ensure models fit on assigned GPUs
- **Parameter validation** for all configuration fields
- **Model database validation** for consistency

Common issues caught:
- Models too large for GPU memory
- Invalid GPU references in parallelism configs
- Missing required fields
- Inconsistent parameter values

See `src/llmperforacle/utils/config_validator.py` for validation implementation.

## Future Extension Points

- ~~Multi-GPU simulation support~~ ✓ Implemented (TP, PP, DP)
- Disaggregated architecture modeling  
- ~~Pipeline/tensor parallelism~~ ✓ Implemented
- Advanced scheduling algorithms
- Real trace replay capability
- Cost-based optimization
- TensorRT-LLM framework adapter
- SGLang with RadixAttention
- Dynama disaggregated serving
- Energy consumption modeling
- Prefix caching simulation
- Speculative decoding
- Continuous batching optimizations