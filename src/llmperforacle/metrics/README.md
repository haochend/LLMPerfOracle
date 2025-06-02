# Metrics Collection and Reporting

This module systematically collects, aggregates, and reports performance metrics during simulation.

## Overview

The Metrics Collector tracks all simulation events and calculates key performance indicators for LLM serving, including latency metrics, throughput, and resource utilization.

## Components

### MetricsCollector

Central aggregator for all simulation metrics.

**Key Features:**
- Per-request metric tracking
- Time-series resource utilization logs
- Statistical aggregation with percentiles
- Export to JSON and CSV formats

### RequestMetricsEntry

Stores comprehensive metrics for each request:
- Timestamps: arrival, dispatch, prefill start, first token, completion
- Token counts: prompt, requested output, actual output
- Status: SUCCESS, FAILURE_OOM_KV, FAILURE_TIMEOUT, ABORTED
- Calculated metrics: TTFT, TPOT, E2E latency
- Prefix caching metrics: cache hits, cached tokens, computation savings
- Chunked prefill metrics: number of chunks, chunk processing times

### TimePointMetric

Records time-series data for resource utilization:
- GPU utilization
- KV cache usage
- Network bandwidth consumption

## Key Metrics

### Latency Metrics
- **Time to First Token (TTFT)**: Time from request arrival/prefill to first token
- **Time Per Output Token (TPOT)**: Average time between tokens after the first
- **End-to-End Latency**: Total request processing time

### Prefix Caching Metrics
- **Cache Hit Rate**: Percentage of requests with prefix cache hits
- **Conversational Hit Rate**: Hit rate for multi-turn conversations
- **Cross-Request Hit Rate**: Hit rate for shared prefixes
- **Tokens Saved**: Number of tokens skipped due to caching
- **Computation Reduction**: Percentage of prefill work avoided

### Throughput Metrics
- **Requests Per Second (RPS)**: Successfully completed requests
- **Output Tokens Per Second (TPS)**: Total token generation rate

### Resource Utilization
- **GPU Utilization**: Percentage of time GPU is busy
- **KV Cache Utilization**: Memory blocks used vs. available
- **Network Utilization**: Bandwidth usage over time

## Logging Methods

```python
# Request lifecycle
metrics.log_request_arrival(request_id, arrival_time, ...)
metrics.log_prefill_start(request_id, start_time)
metrics.log_first_token_generated(request_id, time, prefill_start)
metrics.log_token_decoded(request_id, time, token_count)
metrics.log_request_completed(request_id, time, tokens, status)

# Prefix caching
metrics.log_prefix_cache_event(request_id, event_type, cached_tokens, ...)
# Event types: MISS_FULL, CONVERSATIONAL_HIT, CROSS_REQUEST_HIT

# Chunked prefill
metrics.log_chunked_prefill_progress(request_id, chunk_num, tokens_processed)

# Resource tracking
metrics.log_kv_cache_usage(framework_id, time, used, total)
metrics.log_gpu_task_start(gpu_id, time, task_id)
metrics.log_gpu_task_end(gpu_id, time, task_id)
```

## Output Formats

### Summary Report (JSON)
```json
{
  "simulation": {
    "total_duration_s": 300,
    "warm_up_duration_s": 30,
    "effective_duration_s": 270
  },
  "requests": {
    "total": 1000,
    "successful": 950,
    "failed": 50,
    "success_rate": 0.95
  },
  "throughput": {
    "requests_per_second": 3.52,
    "output_tokens_per_second": 450.3
  },
  "latency": {
    "time_to_first_token_ms": {
      "p50": 45.2,
      "p90": 89.1,
      "p99": 156.3
    }
  },
  "prefix_caching": {
    "overall_hit_rate": 0.71,
    "conversational_hit_rate": 0.94,
    "cross_request_hit_rate": 0.52,
    "tokens_saved": 171000,
    "prefill_computation_reduction": 0.69
  }
}
```

### Detailed Metrics (CSV)
- Per-request data with all timestamps
- Token counts and status
- Individual latency measurements

## Configuration

```yaml
metrics_config:
  percentiles_to_calculate: [0.5, 0.9, 0.95, 0.99]
  warm_up_duration_s: 60  # Exclude initial period
  output_summary_json_path: "results/summary.json"
  output_requests_csv_path: "results/requests.csv"
```