# LLMPerfOracle Simulation Walkthrough

## Overview

This guide walks you through running a simulation and interpreting the results. LLMPerfOracle simulates LLM serving performance without requiring actual hardware.

## Quick Start

```bash
# 1. Activate virtual environment
cd /Users/davidd/LLMPerfOracle
source venv/bin/activate
export PYTHONPATH=/Users/davidd/LLMPerfOracle/src

# 2. Run a simulation
python -m llmperforacle.cli run configs/walkthrough_example.yaml -l INFO

# 3. Check results
cat experiments/results/walkthrough_summary.json
```

## Configuration Components

### 1. Simulation Settings
```yaml
simulation:
  max_simulation_time: 30  # Total simulation duration in seconds
  random_seed: 42         # For reproducible results
```

### 2. Hardware Profile
Defines virtual GPUs and network topology:
- **Compute Devices**: GPU specifications (TFLOPS, memory, bandwidth)
- **Network Links**: Connections between nodes with bandwidth/latency

### 3. Workload Configuration
Specifies request patterns:
- **Client Profiles**: Different user types with arrival patterns
- **Token Distributions**: Input/output token length distributions
- **Conversation Patterns**: Follow-up probability and timing

### 4. Framework Configuration
LLM serving framework settings:
- **Model Selection**: Which model to simulate (e.g., Llama2-13B)
- **Resource Allocation**: GPU assignment, memory limits
- **Batching Parameters**: Max sequences, batch sizes
- **Parallelism**: TP/PP/DP strategies (if enabled)

### 5. Metrics Configuration
Controls output and analysis:
- **Percentiles**: Which latency percentiles to calculate
- **Warm-up Period**: Initial time to exclude from stats
- **Output Paths**: Where to save results

## Understanding the Results

### Summary Report (JSON)

The summary report provides high-level metrics:

#### 1. Simulation Info
```json
"simulation": {
  "total_duration_s": 30,      // Total simulation time
  "warm_up_duration_s": 5,     // Excluded from statistics
  "effective_duration_s": 25   // Used for metric calculation
}
```

#### 2. Request Statistics
```json
"requests": {
  "total": 80,          // Total requests processed
  "successful": 80,     // Successfully completed
  "failed": 0,          // Failed (e.g., timeout, OOM)
  "success_rate": 1.0   // Success percentage
}
```

#### 3. Throughput Metrics
```json
"throughput": {
  "requests_per_second": 3.2,      // Request completion rate
  "output_tokens_per_second": 1627.8  // Token generation rate
}
```

#### 4. Latency Metrics (in milliseconds)

**Time to First Token (TTFT)**:
- Time from request arrival to first output token
- Critical for streaming/interactive applications
```json
"time_to_first_token_ms": {
  "mean": 42.1,    // Average TTFT
  "p50": 17.9,     // Median (50% of requests)
  "p90": 119.0,    // 90th percentile
  "p99": 156.0     // 99th percentile
}
```

**Time Per Output Token (TPOT)**:
- Average time between consecutive tokens
- Indicates generation speed
```json
"time_per_output_token_ms": {
  "mean": 0.106,   // ~9.4 tokens/second per request
  "p50": 0.077,    // Median generation speed
  "p90": 0.153,    // Slower for 10% of tokens
}
```

**End-to-End Latency**:
- Total time from request arrival to completion
```json
"end_to_end_latency_ms": {
  "mean": 104.5,   // Average total latency
  "p50": 59.4,     // Half complete within 59ms
  "p90": 201.0,    // 90% complete within 201ms
  "p99": 339.7     // Worst 1% take up to 340ms
}
```

### Detailed Request Log (CSV)

The CSV file contains per-request metrics:

| Column | Description |
|--------|-------------|
| request_id | Unique request identifier |
| client_id | Which client profile generated it |
| session_id | Conversation session (for multi-turn) |
| prompt_tokens | Input token count |
| max_output_tokens | Requested max output |
| output_tokens | Actual tokens generated |
| arrival_time | When request arrived (simulation time) |
| first_token_time | When first token was generated |
| completion_time | When request completed |
| status | SUCCESS/FAILED/TIMEOUT |
| ttft_ms | Time to first token (ms) |
| tpot_ms | Time per output token (ms) |
| e2e_latency_ms | End-to-end latency (ms) |

## Interpreting Performance

### Good Performance Indicators:
- **High success rate** (>95%)
- **Low TTFT** (<100ms for interactive)
- **Consistent TPOT** (low variance)
- **P99 latency** < 3x P50 latency

### Performance Issues to Watch:
- **High TTFT**: Indicates queuing or slow prefill
- **Variable TPOT**: Suggests resource contention
- **Failed requests**: Memory exhaustion or timeouts
- **P99 >> P90**: Long tail latency problems

### Factors Affecting Performance:

1. **Hardware**:
   - GPU compute capacity (TFLOPS)
   - Memory bandwidth (GB/s)
   - Network latency/bandwidth
   - **Hardware Efficiency** (NEW):
     - MFU (Model FLOPs Utilization): 50% default
     - MBU (Memory Bandwidth Utilization): 80% default

2. **Workload**:
   - Request arrival rate
   - Token length distribution
   - Conversation patterns

3. **Framework Config**:
   - Max batch size
   - Number of KV cache blocks
   - Scheduling policies
   - Parallelism strategy

## Advanced Analysis

### 1. Visualizing Results
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load detailed metrics
df = pd.read_csv('experiments/results/walkthrough_requests.csv')

# Plot TTFT distribution
plt.figure(figsize=(10, 6))
plt.hist(df['ttft_ms'], bins=50, alpha=0.7)
plt.xlabel('Time to First Token (ms)')
plt.ylabel('Request Count')
plt.title('TTFT Distribution')
plt.show()

# Plot request timeline
plt.figure(figsize=(12, 6))
plt.scatter(df['arrival_time'], df['e2e_latency_ms'], alpha=0.5)
plt.xlabel('Arrival Time (s)')
plt.ylabel('End-to-End Latency (ms)')
plt.title('Request Latency Over Time')
plt.show()
```

### 2. Comparing Configurations

Run multiple experiments with different settings:
```bash
# Test different batch sizes
for batch_size in 16 32 64 128; do
  # Modify config and run
  python -m llmperforacle.cli run config_batch_${batch_size}.yaml
done
```

### 3. Analyzing Parallelism

For multi-GPU simulations:
- Compare single GPU vs TP vs PP performance
- Monitor inter-GPU communication overhead
- Analyze load distribution across GPUs

## Common Scenarios

### 1. Chatbot Service
- Focus on TTFT for responsiveness
- Enable streaming responses
- Use smaller batch sizes for lower latency

### 2. Batch Processing
- Optimize for throughput (tokens/second)
- Use larger batch sizes
- Less concern about TTFT

### 3. Mixed Workload
- Balance latency and throughput
- Consider priority scheduling
- Monitor P99 latencies closely

## Troubleshooting

### High Latencies
1. Check if request rate exceeds capacity
2. Increase `max_num_seqs` or `max_num_batched_tokens`
3. Enable parallelism (TP/PP/DP)

### Failed Requests
1. Check KV cache capacity
2. Verify memory calculations
3. Reduce batch size or sequence limits

### Unrealistic Results
1. Verify model parameters match real models
2. Check hardware specifications
3. Validate workload patterns

## Hardware Efficiency Factors

The simulation now includes realistic hardware efficiency factors:

### MFU (Model FLOPs Utilization)
- **Default**: 50% (0.5)
- **Meaning**: LLMs typically achieve only 50% of peak theoretical FLOPs
- **Why**: Memory access patterns, kernel overhead, instruction scheduling

### MBU (Memory Bandwidth Utilization)
- **Default**: 80% (0.8)
- **Meaning**: Memory-bound operations achieve 80% of peak bandwidth
- **Why**: Non-contiguous access patterns, cache effects

### Customizing Efficiency

You can adjust these values in your code:
```python
# In your experiment setup
hardware_platform.set_efficiency_factors(mfu=0.4, mbu=0.75)
```

Or check what efficiency was achieved:
- Look for MFU/MBU percentages in DEBUG logs
- Higher values indicate better hardware utilization

### Impact on Performance

With these efficiency factors:
- **Prefill latency** increases by ~2x (compute-bound, affected by MFU)
- **Decode latency** increases by ~1.25x (memory-bound, affected by MBU)
- Results are more realistic compared to real deployments

## Next Steps

1. **Experiment with different configurations** to understand tradeoffs
2. **Use real workload traces** for more accurate simulations
3. **Compare different parallelism strategies** for large models
4. **Validate against real deployments** when possible
5. **Tune efficiency factors** based on your hardware

The simulation provides insights into:
- Capacity planning
- Configuration optimization  
- Bottleneck identification
- Scaling strategies

Without requiring expensive hardware or long deployment cycles!