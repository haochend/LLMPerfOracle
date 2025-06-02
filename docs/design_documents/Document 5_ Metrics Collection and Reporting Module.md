Document 5: Metrics Collection and Reporting Module
1. Purpose:
To systematically collect, aggregate, and report various performance metrics generated during the simulation. This module provides insights into the behavior and efficiency of the simulated LLM serving frameworks and hardware configurations.
2. Key Data Structures/Classes:
* RequestMetricsEntry (Data Class):
   * Purpose: Stores all relevant timestamps and counters for a single request.
   * Attributes:
      * request_id: str
      * client_id: str
      * session_id: str
      * prompt_num_tokens: int
      * max_output_tokens_requested: int
      * arrival_time_sim: float (simulation time request was logged by workload generator)
      * dispatch_time_sim: float (simulation time request was put into framework queue)
      * prefill_start_time_sim: float (optional, if framework logs it)
      * first_token_emit_time_sim: float (simulation time first output token is ready/sent)
      * completion_time_sim: float (simulation time request processing finishes)
      * output_tokens_generated: int
      * status: str (e.g., "SUCCESS", "FAILURE_OOM_KV", "FAILURE_TIMEOUT", "ABORTED")
      * end_to_end_latency_ms: float (calculated)
      * time_to_first_token_ms: float (calculated)
      * time_per_output_token_ms: float (calculated for successful requests with >1 output token)
      * prefill_duration_ms: float (calculated if prefill_start_time_sim and first_token_emit_time_sim are available)
      * decode_duration_ms: float (calculated if first_token_emit_time_sim and completion_time_sim are available)
* TimePointMetric (Data Class, for time-series data like utilization):
   * timestamp_sim: float
   * value: float
   * resource_id: str (e.g., gpu_id, link_id)
   * metric_type: str (e.g., "GPU_UTILIZATION", "KV_CACHE_USED_BLOCKS")
* MetricsCollector Class:
   * Purpose: Central aggregator for all metrics.
   * Attributes:
      * simpy_env: simpy.Environment (optional, if needing current time directly, but usually time is passed in)
      * all_request_metrics: Dict[str, RequestMetricsEntry] (request_id -> entry)
      * gpu_utilization_log: List[TimePointMetric]
      * kv_cache_usage_log: Dict[str, List[TimePointMetric]] (framework_id -> list of points)
      * network_link_usage_log: Dict[str, List[TimePointMetric]] (link_id -> list of points)
      * config: dict (metrics-specific config, e.g., percentiles)
3. Core Logic/Algorithms:
* MetricsCollector.__init__(self, simpy_env, config: dict):
   1. Stores simpy_env, config.
   2. Initializes all_request_metrics = {} and other log lists/dicts.
* Logging Methods (called by other components):
   * log_request_arrival(self, request_id: str, arrival_time: float, client_id: str, session_id: str, prompt_tokens: int, max_output: int): Creates a new RequestMetricsEntry or updates if partially logged.
   * log_request_dispatch(self, request_id: str, dispatch_time: float): Updates entry.
   * log_prefill_start(self, request_id: str, prefill_start_time: float): Updates entry.
   * log_first_token_generated(self, request_id: str, first_token_time: float, prefill_start_time_for_ttft_calc: float): Updates entry. prefill_start_time_for_ttft_calc is crucial if TTFT is measured from prefill start.
   * log_token_decoded(self, request_id: str, decode_time: float, current_output_token_count: int): Updates output_tokens_generated in entry (or a separate token emission log if very granular data is needed).
   * log_request_completed(self, request_id: str, completion_time: float, final_output_token_count: int, status: str): Updates entry with completion details. Calculates derived latencies (TTFT, TPOT, E2E) here.
   * log_kv_cache_usage(self, framework_id: str, timestamp: float, used_blocks: int, total_blocks: int): Appends TimePointMetric to kv_cache_usage_log[framework_id].
   * log_gpu_task_start(self, gpu_id: str, timestamp: float, task_id: str) / log_gpu_task_end(self, gpu_id: str, timestamp: float, task_id: str): These can be used to infer GPU busy/idle times. Actual utilization calculation happens in reporting.
   * log_network_transfer_start(self, link_id: str, timestamp: float, data_size_bytes: int) / log_network_transfer_end(self, link_id: str, timestamp: float): For inferring link utilization/throughput.
* Metric Calculation (within log_request_completed or in generate_summary_report):
   * entry.end_to_end_latency_ms = (entry.completion_time_sim - entry.arrival_time_sim) * 1000
   * If entry.first_token_emit_time_sim is valid and entry.prefill_start_time_sim is valid (or using entry.arrival_time_sim as alternative start for TTFT):
entry.time_to_first_token_ms = (entry.first_token_emit_time_sim - entry.prefill_start_time_sim) * 1000 (or arrival_time_sim)
   * If entry.output_tokens_generated > 1 and entry.first_token_emit_time_sim is valid:
entry.time_per_output_token_ms = ((entry.completion_time_sim - entry.first_token_emit_time_sim) / (entry.output_tokens_generated - 1)) * 1000
   * entry.prefill_duration_ms = (entry.first_token_emit_time_sim - entry.prefill_start_time_sim) * 1000 (if both available)
   * entry.decode_duration_ms = (entry.completion_time_sim - entry.first_token_emit_time_sim) * 1000 (if both available)
   * generate_summary_report(self, simulation_duration_s: float) -> dict:
   1. Filter out incomplete requests or warm-up period if configured.
   2. Calculate global summary statistics from all_request_metrics.values():
   * Total requests processed, successful, failed.
   * Average, median, P90, P95, P99 for TTFT, TPOT, End-to-End Latency.
   * Overall throughput (successful requests per second, output tokens per second).
   * Goodput (successful output tokens per second).
   3. Calculate average GPU utilization: From gpu_task_start/end logs, calculate total busy time / simulation duration for each GPU.
   4. Calculate average KV cache utilization: Average used_blocks / total_blocks over time for each framework.
   5. Calculate average network link utilization/throughput.
   6. Return a structured dictionary with all these summary metrics.
   7. (Optional) Print a human-readable version of the report.
4. Interfaces/APIs:
   * Constructor: MetricsCollector(simpy_env, config)
   * All log_* methods as defined above.
   * generate_summary_report(simulation_duration_s: float) -> dict
   * get_all_request_metrics_df(self) -> pandas.DataFrame: Returns a DataFrame of all request metrics for detailed analysis (requires pandas).
   * get_timeseries_log_df(self, metric_type: str, resource_id: str = None) -> pandas.DataFrame: Returns a DataFrame for a specific time-series metric.
5. Configuration Parameters (from metrics section of main experiment config):
   * percentiles_to_calculate: List[float] (e.g., [0.5, 0.9, 0.95, 0.99])
   * warm_up_duration_s: float (requests arriving before this time are excluded from summary stats).
   * output_csv_path: str (optional, path to save detailed request metrics).
   * output_summary_path: str (optional, path to save summary report as JSON/text).
6. Dependencies:
   * Fed by: WorkloadGenerator, LLMFrameworkModule implementations, VirtualHardwarePlatform.
   * (Optional) pandas library for DataFrame outputs.