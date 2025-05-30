# Hardware Efficiency Factors (MFU/MBU)

LLMPerfOracle now includes realistic hardware efficiency factors to model the gap between theoretical peak performance and achieved performance in real systems.

## Overview

### MFU (Model FLOPs Utilization)
- **Default**: 50% (0.5)
- **Meaning**: LLMs typically achieve only 50% of peak theoretical FLOPs
- **Why**: Memory access patterns, kernel overhead, instruction scheduling inefficiencies
- **Impact**: Primarily affects compute-bound operations (e.g., prefill)

### MBU (Memory Bandwidth Utilization)
- **Default**: 80% (0.8)
- **Meaning**: Memory-bound operations achieve 80% of peak bandwidth
- **Why**: Non-contiguous access patterns, cache effects, memory controller overhead
- **Impact**: Primarily affects memory-bound operations (e.g., decode)

## Usage

### Default Behavior
The simulation automatically applies these efficiency factors:
```python
# In VirtualHardwarePlatform
MFU_EFFICIENCY = 0.5  # 50%
MBU_EFFICIENCY = 0.8  # 80%
```

### Customizing Efficiency Factors
You can adjust these values for your specific hardware:
```python
# In your experiment setup
hardware_platform.set_efficiency_factors(mfu=0.4, mbu=0.75)
```

### Monitoring Achieved Efficiency
Enable DEBUG logging to see actual MFU/MBU for each operation:
```bash
python -m llmperforacle.cli run config.yaml -l DEBUG
```

Look for log messages like:
```
Task prefill_layer_0 completed... MFU: 50.0%, MBU: 23.0%
Task decode_batch_1 completed... MFU: 87.1%, MBU: 80.0%
```

## Impact on Performance

With these efficiency factors:
- **Prefill latency** increases by ~2x (compute-bound, limited by MFU)
- **Decode latency** increases by ~1.25x (memory-bound, limited by MBU)
- Overall results are more realistic compared to real deployments

## Typical Values

Based on real-world measurements:
- **MFU**: 40-60% for large language models
- **MBU**: 70-90% for streaming memory access
- Higher values indicate better optimization and hardware utilization

## Technical Details

The implementation modifies execution time calculations:
```python
# Compute time with MFU efficiency
effective_tflops = peak_tflops * mfu_efficiency
compute_time = flops_required / effective_tflops

# Memory time with MBU efficiency  
effective_bandwidth = peak_bandwidth * mbu_efficiency
memory_time = bytes_transferred / effective_bandwidth
```

The roofline model then selects the limiting factor:
```python
if is_memory_bound:
    execution_time = memory_time
else:
    execution_time = max(compute_time, memory_time)
```