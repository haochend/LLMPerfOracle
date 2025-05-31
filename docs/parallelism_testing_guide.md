# Parallelism Testing Guide

## Overview

This guide explains how to properly test multi-GPU parallelism features in LLMPerfOracle to demonstrate actual performance benefits.

## Key Problems with Basic Integration Tests

### 1. **Insufficient Load**
- Low request rates (1 req/s) don't stress the system
- Single GPU can handle the load easily
- No benefit from parallelism

### 2. **Unrealistic Workloads**
- Small, uniform request sizes
- No variety in computational requirements
- Missing memory pressure scenarios

### 3. **Poor Network Topology**
- Full mesh unrealistic for 8+ GPUs
- Missing communication overhead
- No consideration of actual hardware limitations

### 4. **Incorrect Success Metrics**
- Expecting linear scaling (unrealistic)
- Not accounting for Amdahl's Law
- Missing overhead considerations

## Improved Testing Approach

### 1. **Heavy Workloads**
```yaml
workload:
  client_profiles:
    - profile_name: "heavy_compute"
      inter_arrival_time_dist_config:
        type: "Exponential"
        rate: 5.0  # 5 req/s minimum
      prompt_tokens_dist_config:
        type: "LogNormal"
        mean: 1500  # Large prompts
        sigma: 500
```

### 2. **Realistic Scenarios**

#### Compute-Bound (Benefits from TP)
- Large prompts (1000-2000 tokens)
- Short outputs (50-200 tokens)
- Tests prefill performance

#### Memory-Bound (Benefits from PP)
- Moderate prompts (200-500 tokens)
- Very long outputs (1000-2000 tokens)
- Tests KV cache management

#### High Throughput (Benefits from DP)
- Many small requests (50+ req/s)
- Mixed sizes
- Tests queuing and scheduling

### 3. **Proper Scaling Expectations**

#### Tensor Parallelism (TP)
- TP=2: Expect 1.5-1.8x speedup
- TP=4: Expect 2.5-3.5x speedup
- Diminishing returns due to communication

#### Pipeline Parallelism (PP)
- Better memory utilization
- Higher request capacity
- Some latency increase

#### Data Parallelism (DP)
- Near-linear throughput scaling
- Consistent latency
- Best for high request rates

### 4. **Realistic Network Topology**
```python
def _generate_realistic_network(num_gpus):
    # Ring topology instead of full mesh
    for i in range(num_gpus):
        next_gpu = (i + 1) % num_gpus
        links.append({
            "link_id": f"gpu{i}_to_gpu{next_gpu}",
            "bandwidth_bps": 600_000_000_000,  # NVLink
            "latency_s": 0.0000005,
            "bidirectional": True
        })
```

## Test Scenarios

### 1. **TP for Compute-Bound Workload**
```python
def test_tp_for_compute_bound_workload():
    # Large prompts that are compute-intensive
    config["workload"]["client_profiles"] = [{
        "prompt_tokens_dist_config": {
            "type": "Constant",
            "value": 2000  # Very large
        },
        "max_output_tokens_dist_config": {
            "type": "Constant",
            "value": 50  # Short outputs
        }
    }]
```

### 2. **PP for Memory-Constrained**
```python
def test_pp_for_memory_constrained():
    # Reduce memory to force constraints
    for device in config["hardware_profile"]["compute_devices"]:
        device["memory_capacity_bytes"] = 40_000_000_000  # 40GB
    
    # Long outputs stress KV cache
    config["workload"]["client_profiles"] = [{
        "max_output_tokens_dist_config": {
            "type": "Constant",
            "value": 1500  # Very long
        }
    }]
```

### 3. **DP for High Throughput**
```python
def test_dp_for_high_throughput():
    # Very high request rate
    config["workload"]["client_profiles"] = [{
        "inter_arrival_time_dist_config": {
            "type": "Exponential",
            "rate": 50.0  # 50 req/s!
        }
    }]
```

## Metrics to Track

### 1. **Throughput Metrics**
- Requests per second
- Tokens per second
- GPU utilization

### 2. **Latency Metrics**
- Time to First Token (TTFT)
- Time Per Output Token (TPOT)
- P50, P90, P99 percentiles

### 3. **Efficiency Metrics**
- Success rate
- Queue depths
- Memory utilization

## Running the Tests

```bash
# Run all improved parallel tests
pytest tests/integration/test_parallel_simulation_improved.py -v

# Run specific scenario
pytest tests/integration/test_parallel_simulation_improved.py::TestRealisticParallelScenarios::test_tp_for_compute_bound_workload -v
```

## Interpreting Results

### Good TP Scaling
```json
{
  "tp1": {"latency": {"time_to_first_token_ms": {"p50": 100}}},
  "tp2": {"latency": {"time_to_first_token_ms": {"p50": 60}}},  // 1.67x speedup
  "tp4": {"latency": {"time_to_first_token_ms": {"p50": 35}}}   // 2.86x speedup
}
```

### Good DP Scaling
```json
{
  "dp1": {"throughput": {"requests_per_second": 12}},
  "dp2": {"throughput": {"requests_per_second": 23}},  // 1.92x
  "dp4": {"throughput": {"requests_per_second": 45}}   // 3.75x
}
```

## Common Pitfalls

1. **Testing with too light load** - System never gets stressed
2. **Expecting perfect scaling** - Communication overhead is real
3. **Wrong parallelism for workload** - TP for memory-bound won't help
4. **Ignoring warm-up period** - Initial results are noisy
5. **Not checking GPU utilization** - Low utilization means wasted resources

## Best Practices

1. **Match parallelism to bottleneck**:
   - Compute-bound → Tensor Parallelism
   - Memory-bound → Pipeline Parallelism
   - Throughput-bound → Data Parallelism

2. **Use realistic workloads**:
   - Mixed request sizes
   - Bursty arrival patterns
   - Long-running simulations (60+ seconds)

3. **Monitor all metrics**:
   - Not just latency or throughput
   - Check utilization and efficiency
   - Look for bottlenecks

4. **Test incrementally**:
   - Start with single GPU baseline
   - Add parallelism gradually
   - Compare each configuration

The improved tests in `test_parallel_simulation_improved.py` follow these best practices and demonstrate real benefits from parallelism.