# Experiment Configuration and Orchestration

This module manages the setup and execution of simulation experiments.

## Overview

The Experiment Orchestrator is the main entry point that:
- Loads and validates configuration
- Initializes all simulation components
- Coordinates module interactions
- Manages experiment execution
- Generates final reports

## Components

### ExperimentOrchestrator

The central coordinator for simulation experiments.

**Key Responsibilities:**
- Configuration parsing and validation (using ExperimentConfigValidator)
- Component initialization in correct order
- Framework instance creation
- Simulation execution management
- Result collection and reporting
- Memory validation to prevent invalid configurations

## Configuration Structure

The orchestrator expects a comprehensive configuration covering all aspects:

```yaml
# Simulation settings
simulation:
  max_simulation_time: 300
  random_seed: 42

# Model database path
model_characteristics_db_path: "./configs/model_params.json"

# Hardware configuration
hardware_profile:
  compute_devices: [...]
  network_links: [...]

# Workload configuration
workload:
  total_duration: 300
  client_profiles: [...]

# Frameworks to test
frameworks_to_test:
  - name: "vllm_instance"
    type: "VLLM"
    is_target_for_workload: true
    config: {...}

# Metrics configuration
metrics_config:
  percentiles_to_calculate: [0.5, 0.9, 0.95, 0.99]
  warm_up_duration_s: 30
  output_summary_json_path: "results/summary.json"
  output_requests_csv_path: "results/requests.csv"
```

## Initialization Order

1. **Core Simulation Engine**: SimPy environment setup
2. **Metrics Collector**: For tracking all events
3. **Virtual Hardware Platform**: Hardware resource initialization
4. **LLM Framework Instances**: Create framework simulations
5. **Workload Generator**: Configure request generation

## Usage

### From Configuration File

```python
# From YAML
orchestrator = ExperimentOrchestrator.from_yaml_file("config.yaml")
summary = orchestrator.run()

# From JSON
orchestrator = ExperimentOrchestrator.from_json_file("config.json")
summary = orchestrator.run()
```

### Programmatic Configuration

```python
config = {
    "simulation": {...},
    "hardware_profile": {...},
    "workload": {...},
    "frameworks_to_test": [...],
    "metrics_config": {...}
}

orchestrator = ExperimentOrchestrator(config)
summary = orchestrator.run()
```

## Framework Registration

New frameworks are registered in `FRAMEWORK_CLASS_MAP`:

```python
FRAMEWORK_CLASS_MAP = {
    "VLLM": VLLMFramework,
    "ParallelVLLM": ParallelVLLMFramework,
    "TRTLLM": TRTLLMFramework,  # Future
    "SGLang": SGLangFramework,  # Future
    # Add new frameworks here
}
```

## Output

The orchestrator produces:
- Summary statistics (JSON)
- Detailed per-request metrics (CSV)
- Console output with key results

## Error Handling

The orchestrator validates:
- Required configuration sections
- Model profile availability
- Framework type recognition
- Hardware resource definitions