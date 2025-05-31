"""Metrics collection and reporting implementation."""

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import simpy

from .models import RequestMetricsEntry, TimePointMetric

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Central aggregator for all simulation metrics."""
    
    def __init__(self, simpy_env: simpy.Environment, config: Dict[str, Any]):
        """Initialize the metrics collector.
        
        Args:
            simpy_env: SimPy environment for accessing simulation time
            config: Metrics configuration containing:
                - percentiles_to_calculate: List of percentiles (e.g., [0.5, 0.9, 0.99])
                - warm_up_duration_s: Warm-up period to exclude from stats
                - output_csv_path: Path to save detailed metrics
                - output_summary_path: Path to save summary report
        """
        self.simpy_env = simpy_env
        self.config = config
        
        # Request metrics storage
        self.all_request_metrics: Dict[str, RequestMetricsEntry] = {}
        
        # Time-series logs
        self.gpu_utilization_log: List[TimePointMetric] = []
        self.kv_cache_usage_log: Dict[str, List[TimePointMetric]] = {}
        self.network_link_usage_log: Dict[str, List[TimePointMetric]] = {}
        
        # GPU task tracking for utilization calculation
        self.gpu_task_log: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info("MetricsCollector initialized")
    
    def log_request_arrival(
        self,
        request_id: str,
        arrival_time: float,
        client_id: str,
        session_id: str,
        prompt_tokens: int,
        max_output: int,
    ) -> None:
        """Log the arrival of a new request."""
        entry = RequestMetricsEntry(
            request_id=request_id,
            client_id=client_id,
            session_id=session_id,
            prompt_num_tokens=prompt_tokens,
            max_output_tokens_requested=max_output,
            arrival_time_sim=arrival_time,
        )
        self.all_request_metrics[request_id] = entry
        logger.debug(f"Logged arrival of request {request_id} at time {arrival_time}")
    
    def log_request_dispatch(self, request_id: str, dispatch_time: float) -> None:
        """Log when a request is dispatched to a framework."""
        if request_id in self.all_request_metrics:
            self.all_request_metrics[request_id].dispatch_time_sim = dispatch_time
    
    def log_prefill_start(self, request_id: str, prefill_start_time: float, num_tokens_actually_prefilled: Optional[int] = None) -> None:
        """Log when prefill processing starts for a request."""
        if request_id in self.all_request_metrics:
            entry = self.all_request_metrics[request_id]
            entry.prefill_start_time_sim = prefill_start_time
            if num_tokens_actually_prefilled is not None:
                entry.num_tokens_actually_prefilled = num_tokens_actually_prefilled
    
    def log_first_token_generated(
        self, request_id: str, first_token_time: float, prefill_start_time: float
    ) -> None:
        """Log when the first output token is generated."""
        if request_id in self.all_request_metrics:
            entry = self.all_request_metrics[request_id]
            entry.first_token_emit_time_sim = first_token_time
            # Also update prefill start time if not already set
            if entry.prefill_start_time_sim is None:
                entry.prefill_start_time_sim = prefill_start_time
    
    def log_token_decoded(
        self, request_id: str, decode_time: float, current_output_token_count: int
    ) -> None:
        """Log token generation progress."""
        if request_id in self.all_request_metrics:
            self.all_request_metrics[request_id].output_tokens_generated = current_output_token_count
    
    def log_request_completed(
        self, request_id: str, completion_time: float, final_output_token_count: int, status: str
    ) -> None:
        """Log request completion and calculate derived metrics."""
        if request_id not in self.all_request_metrics:
            logger.warning(f"Request {request_id} not found in metrics")
            return
        
        entry = self.all_request_metrics[request_id]
        entry.completion_time_sim = completion_time
        entry.output_tokens_generated = final_output_token_count
        entry.status = status
        
        # Calculate derived metrics (convert to milliseconds)
        entry.end_to_end_latency_ms = (completion_time - entry.arrival_time_sim) * 1000
        
        if entry.first_token_emit_time_sim is not None:
            # Time to first token (from arrival or prefill start)
            ttft_start = entry.prefill_start_time_sim or entry.arrival_time_sim
            entry.time_to_first_token_ms = (entry.first_token_emit_time_sim - ttft_start) * 1000
            
            # Time per output token (for tokens after the first)
            if entry.output_tokens_generated > 1:
                entry.time_per_output_token_ms = (
                    (completion_time - entry.first_token_emit_time_sim) /
                    (entry.output_tokens_generated - 1)
                ) * 1000
            
            # Prefill and decode durations
            if entry.prefill_start_time_sim is not None:
                entry.prefill_duration_ms = (
                    entry.first_token_emit_time_sim - entry.prefill_start_time_sim
                ) * 1000
            
            entry.decode_duration_ms = (
                completion_time - entry.first_token_emit_time_sim
            ) * 1000
        
        logger.debug(
            f"Request {request_id} completed: status={status}, "
            f"tokens={final_output_token_count}, latency={entry.end_to_end_latency_ms:.1f}ms"
        )
    
    def log_kv_cache_usage(
        self, framework_id: str, timestamp: float, used_blocks: int, total_blocks: int
    ) -> None:
        """Log KV cache utilization for a framework."""
        if framework_id not in self.kv_cache_usage_log:
            self.kv_cache_usage_log[framework_id] = []
        
        metric = TimePointMetric(
            timestamp_sim=timestamp,
            value=used_blocks / total_blocks if total_blocks > 0 else 0,
            resource_id=framework_id,
            metric_type="KV_CACHE_UTILIZATION",
        )
        self.kv_cache_usage_log[framework_id].append(metric)
    
    def log_gpu_task_start(self, gpu_id: str, timestamp: float, task_id: str) -> None:
        """Log the start of a GPU task."""
        if gpu_id not in self.gpu_task_log:
            self.gpu_task_log[gpu_id] = []
        
        self.gpu_task_log[gpu_id].append({
            "task_id": task_id,
            "start_time": timestamp,
            "end_time": None,
        })
    
    def log_gpu_task_end(self, gpu_id: str, timestamp: float, task_id: str) -> None:
        """Log the end of a GPU task."""
        if gpu_id in self.gpu_task_log:
            # Find the matching task and update end time
            for task in reversed(self.gpu_task_log[gpu_id]):
                if task["task_id"] == task_id and task["end_time"] is None:
                    task["end_time"] = timestamp
                    break
    
    def log_network_transfer_start(
        self, link_id: str, timestamp: float, data_size_bytes: int
    ) -> None:
        """Log the start of a network transfer."""
        if link_id not in self.network_link_usage_log:
            self.network_link_usage_log[link_id] = []
        
        metric = TimePointMetric(
            timestamp_sim=timestamp,
            value=data_size_bytes,
            resource_id=link_id,
            metric_type="NETWORK_TRANSFER_START",
        )
        self.network_link_usage_log[link_id].append(metric)
    
    def log_network_transfer_end(self, link_id: str, timestamp: float) -> None:
        """Log the end of a network transfer."""
        if link_id not in self.network_link_usage_log:
            self.network_link_usage_log[link_id] = []
        
        metric = TimePointMetric(
            timestamp_sim=timestamp,
            value=0,
            resource_id=link_id,
            metric_type="NETWORK_TRANSFER_END",
        )
        self.network_link_usage_log[link_id].append(metric)
    
    def log_prefix_cache_event(
        self, 
        request_id: str, 
        timestamp: float, 
        event_type: str, 
        cached_prefix_length: int,
        num_tokens_prefilled_after_cache_check: int
    ) -> None:
        """Log prefix cache events."""
        if request_id in self.all_request_metrics:
            entry = self.all_request_metrics[request_id]
            # Store prefix cache info
            entry.prefix_cache_event_type = event_type
            entry.cached_prefix_length_used = cached_prefix_length
            entry.num_tokens_actually_prefilled = num_tokens_prefilled_after_cache_check
            
            logger.debug(
                f"Prefix cache {event_type} for {request_id}: "
                f"cached={cached_prefix_length}, to_prefill={num_tokens_prefilled_after_cache_check}"
            )
    
    def generate_summary_report(self, simulation_duration_s: float) -> Dict[str, Any]:
        """Generate comprehensive summary statistics.
        
        Args:
            simulation_duration_s: Total simulation duration in seconds
            
        Returns:
            Dictionary containing all summary metrics
        """
        warm_up_duration = self.config.get("warm_up_duration_s", 0)
        percentiles = self.config.get("percentiles_to_calculate", [0.5, 0.9, 0.95, 0.99])
        
        # Filter out warm-up period requests
        filtered_requests = {
            req_id: entry for req_id, entry in self.all_request_metrics.items()
            if entry.arrival_time_sim >= warm_up_duration
        }
        
        # Separate successful and failed requests
        successful_requests = [
            entry for entry in filtered_requests.values()
            if entry.status == "SUCCESS"
        ]
        failed_requests = [
            entry for entry in filtered_requests.values()
            if entry.status != "SUCCESS"
        ]
        
        # Calculate latency statistics
        ttft_values = [
            entry.time_to_first_token_ms for entry in successful_requests
            if entry.time_to_first_token_ms is not None
        ]
        tpot_values = [
            entry.time_per_output_token_ms for entry in successful_requests
            if entry.time_per_output_token_ms is not None
        ]
        e2e_values = [
            entry.end_to_end_latency_ms for entry in successful_requests
            if entry.end_to_end_latency_ms is not None
        ]
        
        # Calculate throughput
        total_output_tokens = sum(
            entry.output_tokens_generated for entry in successful_requests
        )
        effective_duration = simulation_duration_s - warm_up_duration
        
        summary = {
            "simulation": {
                "total_duration_s": simulation_duration_s,
                "warm_up_duration_s": warm_up_duration,
                "effective_duration_s": effective_duration,
            },
            "requests": {
                "total": len(filtered_requests),
                "successful": len(successful_requests),
                "failed": len(failed_requests),
                "success_rate": len(successful_requests) / len(filtered_requests) if filtered_requests else 0,
            },
            "throughput": {
                "requests_per_second": len(successful_requests) / effective_duration if effective_duration > 0 else 0,
                "output_tokens_per_second": total_output_tokens / effective_duration if effective_duration > 0 else 0,
            },
            "latency": {
                "time_to_first_token_ms": self._calculate_stats(ttft_values, percentiles),
                "time_per_output_token_ms": self._calculate_stats(tpot_values, percentiles),
                "end_to_end_latency_ms": self._calculate_stats(e2e_values, percentiles),
            },
            "gpu_utilization": self._calculate_gpu_utilization(simulation_duration_s),
        }
        
        # Add prefix caching metrics
        prefix_cache_stats = self._calculate_prefix_cache_stats(filtered_requests)
        if prefix_cache_stats:
            summary["prefix_caching"] = prefix_cache_stats
        
        # Log summary
        logger.info("=" * 60)
        logger.info("SIMULATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Duration: {simulation_duration_s:.1f}s (warm-up: {warm_up_duration}s)")
        logger.info(f"Requests: {summary['requests']['total']} total, "
                   f"{summary['requests']['successful']} successful "
                   f"({summary['requests']['success_rate']:.1%} success rate)")
        logger.info(f"Throughput: {summary['throughput']['requests_per_second']:.2f} req/s, "
                   f"{summary['throughput']['output_tokens_per_second']:.1f} tokens/s")
        logger.info(f"TTFT (ms): P50={summary['latency']['time_to_first_token_ms'].get('p50', 0):.1f}, "
                   f"P99={summary['latency']['time_to_first_token_ms'].get('p99', 0):.1f}")
        logger.info("=" * 60)
        
        return summary
    
    def _calculate_stats(self, values: List[float], percentiles: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of values."""
        if not values:
            return {"count": 0}
        
        stats = {
            "count": len(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
        }
        
        # Add percentiles
        for p in percentiles:
            stats[f"p{int(p * 100)}"] = np.percentile(values, p * 100)
        
        return stats
    
    def _calculate_gpu_utilization(self, simulation_duration_s: float) -> Dict[str, float]:
        """Calculate GPU utilization from task logs."""
        gpu_utilization = {}
        
        for gpu_id, tasks in self.gpu_task_log.items():
            total_busy_time = 0
            for task in tasks:
                if task["end_time"] is not None:
                    total_busy_time += task["end_time"] - task["start_time"]
            
            utilization = total_busy_time / simulation_duration_s if simulation_duration_s > 0 else 0
            gpu_utilization[gpu_id] = utilization
        
        return gpu_utilization
    
    def get_all_request_metrics_df(self) -> pd.DataFrame:
        """Get all request metrics as a pandas DataFrame."""
        if not self.all_request_metrics:
            return pd.DataFrame()
        
        # Convert to list of dicts for DataFrame
        metrics_list = []
        for entry in self.all_request_metrics.values():
            metrics_list.append({
                "request_id": entry.request_id,
                "client_id": entry.client_id,
                "session_id": entry.session_id,
                "prompt_tokens": entry.prompt_num_tokens,
                "max_output_tokens": entry.max_output_tokens_requested,
                "output_tokens": entry.output_tokens_generated,
                "arrival_time": entry.arrival_time_sim,
                "first_token_time": entry.first_token_emit_time_sim,
                "completion_time": entry.completion_time_sim,
                "status": entry.status,
                "ttft_ms": entry.time_to_first_token_ms,
                "tpot_ms": entry.time_per_output_token_ms,
                "e2e_latency_ms": entry.end_to_end_latency_ms,
                "prefix_cache_event": getattr(entry, 'prefix_cache_event_type', None),
                "cached_prefix_length": getattr(entry, 'cached_prefix_length_used', 0),
                "tokens_actually_prefilled": getattr(entry, 'num_tokens_actually_prefilled', None),
            })
        
        return pd.DataFrame(metrics_list)
    
    def get_timeseries_log_df(
        self, metric_type: str, resource_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Get time-series metrics as a pandas DataFrame."""
        metrics_list = []
        
        # Collect metrics based on type
        if metric_type == "KV_CACHE_UTILIZATION":
            for fw_id, metrics in self.kv_cache_usage_log.items():
                if resource_id is None or resource_id == fw_id:
                    for metric in metrics:
                        metrics_list.append({
                            "timestamp": metric.timestamp_sim,
                            "value": metric.value,
                            "resource_id": metric.resource_id,
                            "metric_type": metric.metric_type,
                        })
        
        return pd.DataFrame(metrics_list)
    
    def _calculate_prefix_cache_stats(self, requests: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate prefix caching statistics."""
        if not requests:
            return None
        
        # Count cache events by type
        event_counts = {
            "CONVERSATIONAL_HIT": 0,
            "CROSS_REQUEST_HIT": 0,
            "MISS_FULL": 0,
            "FULL_HIT_NO_PREFILL_NEEDED": 0,
            "CONVERSATIONAL_MISS_UNEXPECTED_PROMPT": 0,
        }
        
        total_cached_tokens = 0
        total_prefilled_tokens = 0
        total_prompt_tokens = 0
        conversational_requests = 0
        
        for entry in requests.values():
            # Count event types
            if hasattr(entry, 'prefix_cache_event_type') and entry.prefix_cache_event_type:
                event_type = entry.prefix_cache_event_type
                if event_type in event_counts:
                    event_counts[event_type] += 1
            
            # Track token counts
            total_prompt_tokens += entry.prompt_num_tokens
            if hasattr(entry, 'cached_prefix_length_used'):
                total_cached_tokens += entry.cached_prefix_length_used
            if hasattr(entry, 'num_tokens_actually_prefilled') and entry.num_tokens_actually_prefilled is not None:
                total_prefilled_tokens += entry.num_tokens_actually_prefilled
            
            # Count conversational requests
            if entry.session_id:
                conversational_requests += 1
        
        # Calculate hit rates
        total_events = sum(event_counts.values())
        hits = event_counts["CONVERSATIONAL_HIT"] + event_counts["CROSS_REQUEST_HIT"] + event_counts["FULL_HIT_NO_PREFILL_NEEDED"]
        
        stats = {
            "overall_hit_rate": hits / total_events if total_events > 0 else 0,
            "conversational_hit_rate": event_counts["CONVERSATIONAL_HIT"] / conversational_requests if conversational_requests > 0 else 0,
            "event_counts": event_counts,
            "average_cached_prefix_length": total_cached_tokens / hits if hits > 0 else 0,
            "average_tokens_prefilled": total_prefilled_tokens / len(requests) if requests else 0,
            "prefill_reduction_ratio": 1 - (total_prefilled_tokens / total_prompt_tokens) if total_prompt_tokens > 0 else 0,
            "total_tokens_saved": total_cached_tokens,
        }
        
        # Log prefix caching summary
        if stats["overall_hit_rate"] > 0:
            logger.info(f"Prefix Cache Hit Rate: {stats['overall_hit_rate']:.1%}")
            logger.info(f"Prefill Reduction: {stats['prefill_reduction_ratio']:.1%}")
            logger.info(f"Total Tokens Saved: {stats['total_tokens_saved']:,}")
        
        return stats