"""Data models for metrics collection."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RequestMetricsEntry:
    """Stores all relevant timestamps and counters for a single request."""
    
    request_id: str
    client_id: str
    session_id: str
    prompt_num_tokens: int
    max_output_tokens_requested: int
    
    # Timestamps (in simulation time)
    arrival_time_sim: float
    dispatch_time_sim: Optional[float] = None
    prefill_start_time_sim: Optional[float] = None
    first_token_emit_time_sim: Optional[float] = None
    completion_time_sim: Optional[float] = None
    
    # Results
    output_tokens_generated: int = 0
    status: str = "PENDING"  # SUCCESS, FAILURE_OOM_KV, FAILURE_TIMEOUT, ABORTED
    
    # Calculated metrics (in milliseconds)
    end_to_end_latency_ms: Optional[float] = None
    time_to_first_token_ms: Optional[float] = None
    time_per_output_token_ms: Optional[float] = None
    prefill_duration_ms: Optional[float] = None
    decode_duration_ms: Optional[float] = None


@dataclass
class TimePointMetric:
    """Time-series data point for resource utilization metrics."""
    
    timestamp_sim: float
    value: float
    resource_id: str  # e.g., gpu_id, link_id
    metric_type: str  # e.g., "GPU_UTILIZATION", "KV_CACHE_USED_BLOCKS"