# Configuration Validation System

## Overview

To prevent configuration-related issues that we've encountered during development, we've implemented a comprehensive configuration validation system. This system validates:

- Model configurations
- Hardware configurations  
- Framework configurations
- Complete experiment configurations

## Key Features

### 1. Model Configuration Validation

The validator ensures all models have:
- Required fields (parameters, hidden_size, num_layers, etc.)
- `parameter_bytes_fp16` field (auto-calculated if missing)
- Consistent parameter sizes
- Required operation statistics
- KV cache configuration

Example issues caught:
- Missing `parameter_bytes_fp16` field
- Inconsistent parameter count vs bytes
- Missing operation statistics for LoD support

### 2. Hardware Configuration Validation

The validator checks:
- All devices have required fields
- No duplicate device IDs
- Realistic memory capacities (> 1GB)
- Valid network links (devices exist)
- Proper peak_tflops format

Example issues caught:
- Network links referencing non-existent devices
- Duplicate GPU IDs
- Missing required device fields

### 3. Framework Configuration Validation

The validator ensures:
- Model profiles exist in the database
- GPU assignments are valid
- Parallelism configurations are correct
- GPU counts match parallelism strategy

Example issues caught:
- Assigning models to non-existent GPUs
- TP+PP configurations with wrong GPU count
- References to undefined models

### 4. Memory Validation

During framework initialization, the system also checks:
- Model fits in available GPU memory (90% threshold)
- Proper error messages for oversized models

Example: Llama3-70B (140GB) fails on single 80GB GPU with clear error message

## Usage

### CLI Validation Command

Validate configurations before running:

```bash
# Validate configuration file
llmperforacle validate config.yaml

# Also validate model database
llmperforacle validate config.yaml -m

# Future: Auto-fix common issues
llmperforacle validate config.yaml --fix
```

### Automatic Validation

The validator runs automatically when:
- Loading configurations in ExperimentOrchestrator
- Initializing frameworks (memory checks)

### Python API

```python
from llmperforacle.utils.config_validator import (
    ExperimentConfigValidator,
    ModelConfigValidator,
    validate_and_fix_config
)

# Validate complete configuration
is_valid, errors = ExperimentConfigValidator.validate(config_dict)

# Validate and load from file
is_valid, errors, config = validate_and_fix_config("config.yaml")
```

## Common Issues and Solutions

### 1. Missing parameter_bytes_fp16

**Issue**: Models missing FP16 size specification  
**Solution**: Validator auto-calculates as `parameters * 2`

### 2. Framework Entry Point Links

**Issue**: Network links to "framework_entry_0" (legacy pattern)  
**Solution**: Update to point to actual GPU device IDs

### 3. Memory Overflow

**Issue**: Model too large for assigned GPU  
**Solution**: Use Pipeline Parallelism or larger GPUs

### 4. Parallelism GPU Count

**Issue**: TP+PP configurations with incorrect GPU count  
**Solution**: Ensure `gpu_count = tp_degree * pp_stages`

## Benefits

1. **Early Error Detection**: Catches configuration issues before simulation
2. **Clear Error Messages**: Specific descriptions of what's wrong
3. **Auto-fixes**: Can automatically fix some common issues
4. **Consistency**: Ensures all configurations follow the same schema
5. **Memory Safety**: Prevents OOM errors from oversized models

## Implementation Details

The validation system consists of:

- `ModelConfigValidator`: Validates model parameter database
- `HardwareConfigValidator`: Validates hardware profiles
- `FrameworkConfigValidator`: Validates framework configurations
- `ExperimentConfigValidator`: Orchestrates complete validation

Each validator:
- Returns list of specific error messages
- Can auto-fix certain issues (e.g., missing fields)
- Provides warnings for non-critical issues

## Future Enhancements

1. **Schema Definition**: JSON Schema for formal validation
2. **Auto-fix Mode**: Automatically correct more issues
3. **Configuration Migration**: Update old configs to new format
4. **Interactive Mode**: Guide users through fixing issues
5. **Presets**: Validated configuration templates