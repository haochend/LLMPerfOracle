Document 3: Configurable Workload Generator
1. Purpose:
To generate a stream of LLM service requests that mimic realistic usage patterns, based on configurable statistical distributions and client profiles. These requests are then dispatched to the simulated LLM serving frameworks.
2. Key Data Structures/Classes:
* Request (Data Class):
   * Attributes:
      * request_id: str (unique)
      * client_id: str
      * session_id: str (for conversational context)
      * arrival_time: float (simulation time of arrival)
      * prompt_str: str (optional, for realism/logging, not used in perf calculation)
      * prompt_num_tokens: int
      * max_output_tokens: int
      * is_conversational_turn: bool (True if part of an ongoing session)
      * streaming_response: bool (True if tokens should be streamed back)
      * user_priority: int (Optional, for priority scheduling)
* DistributionSampler (Utility Class):
   * Purpose: Provides methods to sample values from various statistical distributions.
   * Methods:
      * sample(distribution_config: dict) -> float | int:
         * distribution_config: e.g., {'type': 'Exponential', 'rate': 5.0} or {'type': 'LogNormal', 'mean': 200, 'sigma': 50, 'is_int': True}.
         * Logic: Uses random or numpy.random to generate samples based on type and parameters. Handles casting to int if is_int is true.
         * Supported types: Exponential, Uniform, Normal, LogNormal, Pareto, Constant, Weibull, Gamma.
* ClientProfile (Data Class):
   * Purpose: Defines the behavior and request characteristics for a specific type of client or workload.
   * Attributes:
      * profile_name: str
      * weight: float (for selecting this profile if multiple exist)
      * inter_arrival_time_dist_config: dict (config for DistributionSampler)
      * prompt_tokens_dist_config: dict
      * max_output_tokens_dist_config: dict
      * conversational_probability: float (0.0 to 1.0)
      * streaming_response_probability: float (0.0 to 1.0)
      * user_priority_dist_config: dict (Optional)
      * (If conversational) follow_up_inter_arrival_time_dist_config: dict (time between turns in a session).
* WorkloadGenerator Class (SimPy Process):
   * Purpose: The main class that generates and dispatches requests.
   * Attributes:
      * simpy_env: simpy.Environment
      * config: dict (workload-specific configuration)
      * client_profiles: List[ClientProfile]
      * sampler: DistributionSampler
      * target_framework_queues: Dict[str, simpy.Store] (framework_id -> queue for requests) or target_framework_handlers: List[AbstractLLMFramework]
      * metrics_collector: MetricsCollector
      * hardware_platform: VirtualHardwarePlatform
      * request_counter: int
      * active_sessions: Dict[str, Dict] (session_id -> last_request_details, next_turn_time)
3. Core Logic/Algorithms:
* WorkloadGenerator.__init__(self, simpy_env, config, target_frameworks_map, metrics_collector, hardware_platform):
   1. Stores dependencies.
   2. Initializes sampler = DistributionSampler().
   3. Parses config['client_profiles'] to create ClientProfile objects.
   4. request_counter = 0.
   5. active_sessions = {}.
* WorkloadGenerator.generate_requests_process(self) -> simpy.Process:
   * This is the main SimPy process loop.
   1. Loop until simpy_env.now < config['total_duration'] (or request count limit).
   2. Select Client Profile: If multiple profiles, select one based on their weights.
   3. Determine Inter-Arrival Time (IAT):
      * Check active_sessions for any client ready for a follow-up conversational turn based on follow_up_inter_arrival_time_dist_config.
      * If a conversational follow-up is due and its time is now or past, prioritize it.
      * Otherwise, sample IAT from the selected client_profile.inter_arrival_time_dist_config.
   4. yield self.simpy_env.timeout(iat_value).
   5. Create Request:
      * self.request_counter += 1. request_id = f"req_{self.request_counter}".
      * client_id (can be derived from profile or generated).
      * Sample prompt_num_tokens and max_output_tokens using sampler and profile's configs.
      * Determine is_conversational_turn, session_id.
         * If new session: session_id = f"sess_{self.request_counter}". is_conversational_turn = False.
         * If follow-up: Use existing session_id. is_conversational_turn = True.
      * Sample streaming_response probability.
      * Sample user_priority.
      * arrival_time = self.simpy_env.now.
      * Create Request object.
   6. Simulate Client-to-Server Network:
      * prompt_data_size_bytes = prompt_num_tokens * BYTES_PER_TOKEN_ESTIMATE (e.g., 2-4 bytes).
      * yield self.simpy_env.process(self.hardware_platform.submit_network_transfer_task("client_node_0", "framework_entry_0", prompt_data_size_bytes))
      * (Note: client_node_0 and framework_entry_0 would be defined in the hardware network config).
   7. Dispatch Request:
      * Select target framework (e.g., round-robin, or based on config).
      * If using queues: target_framework_queues[target_fw_id].put(request).
      * If using direct handlers: self.simpy_env.process(target_framework_handlers[target_idx].handle_incoming_request(request)).
   8. self.metrics_collector.log_request_arrival(request.request_id, request.arrival_time, request.prompt_num_tokens).
   9. Manage Conversational State:
      * If request.is_conversational_turn is false and random.random() < client_profile.conversational_probability:
         * Add session_id to active_sessions and schedule its next potential turn time.
      * If request.is_conversational_turn is true and this is the last turn (e.g., based on a max turns per session or if the model indicates EOS for the session):
         * Remove session_id from active_sessions.
      * Else (if conversational turn and not last):
         * Update next potential turn time for this session in active_sessions.
4. Interfaces/APIs:
* Constructor: WorkloadGenerator(simpy_env, config, target_frameworks_map, metrics_collector, hardware_platform)
* Main Process Initiator: The ExperimentOrchestrator will call simpy_env.process(workload_generator.generate_requests_process()).
5. Configuration Parameters (from workload section of main experiment config):
* total_duration: float or total_requests_to_generate: int.
* client_profiles: List[ClientProfile] (as defined in Key Data Structures).
* bytes_per_token_estimate_for_network: int (e.g., 2).
* random_seed: int.
6. Dependencies:
* Core Simulation Engine (for simpy_env).
* LLM Framework Modules (as targets for dispatched requests).
* Metrics Collector (to log request arrivals).
* Parameterized Virtual Hardware Layer (to simulate client-server network latency/bandwidth for prompt submission).