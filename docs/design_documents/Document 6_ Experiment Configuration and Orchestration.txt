Document 6: Experiment Configuration and Orchestration
1. Purpose:
To define, load, and manage the execution of a complete simulation experiment. This module is responsible for initializing all other components based on a master configuration file/object and running the simulation.
2. Key Data Structures/Classes:
* ExperimentConfig (Data Class or validated Dict):
   * Purpose: Holds the entire configuration for a single simulation experiment. This is typically loaded from a YAML or JSON file.
   * Structure (as outlined in previous thought process, with sections for simulation, hardware_profile, model_characteristics_db_path, workload, frameworks_to_test, metrics_config).
* ExperimentOrchestrator Class:
   * Purpose: The main entry point to set up and run an experiment.
   * Attributes:
      * config: ExperimentConfig
      * sim_env_wrapper: SimulationEnvironment
      * virtual_hardware: VirtualHardwarePlatform
      * metrics_collector: MetricsCollector
      * workload_generator: WorkloadGenerator
      * llm_framework_instances: List[AbstractLLMFramework]
      * model_characteristics_db: Dict
3. Core Logic/Algorithms:
* ExperimentOrchestrator.__init__(self, config_data: dict):
   1. Validates and stores config_data as self.config. (Consider using a library like Pydantic for robust validation of the config structure).
   2. Loads the model characteristics database specified in config.model_characteristics_db_path into self.model_characteristics_db.
* ExperimentOrchestrator.setup_simulation(self):
   1. Initialize Core Simulation Engine:
      * self.sim_env_wrapper = SimulationEnvironment(self.config.simulation)
      * simpy_env_instance = self.sim_env_wrapper.get_simpy_env()
   2. Initialize Metrics Collector:
      * self.metrics_collector = MetricsCollector(simpy_env_instance, self.config.metrics_config)
   3. Initialize Virtual Hardware Platform:
      * self.virtual_hardware = VirtualHardwarePlatform()
      * self.virtual_hardware.initialize(self.config.hardware_profile, simpy_env_instance, self.model_characteristics_db)
   4. Initialize LLM Framework Instances:
      * self.llm_framework_instances = []
      * For each fw_config_entry in self.config.frameworks_to_test: (changed from fw_config to avoid conflict)
         * Get framework_type = fw_config_entry.type (e.g., "VLLM").
         * Dynamically find and instantiate the corresponding class. This requires a mapping or naming convention.
# Example class mapping (to be defined elsewhere, e.g., in a factory or registry)
# from .frameworks import VLLMFramework, TRTLLMFramework # Example imports
# framework_class_map = {"VLLM": VLLMFramework, "TRTLLM": TRTLLMFramework}
FrameworkClass = framework_class_map.get(framework_type)
if not FrameworkClass:
   raise ValueError(f"Unknown framework type: {framework_type}")

         * model_profile_data = self.model_characteristics_db.get(fw_config_entry.config.model_profile_id)
         * If model_profile_data is None:
raise ValueError(f"Model profile ID {fw_config_entry.config.model_profile_id} not found in database for framework {fw_config_entry.name}")
         * instance = FrameworkClass(framework_id=fw_config_entry.name, simpy_env=simpy_env_instance, framework_specific_config=fw_config_entry.config, virtual_hardware=self.virtual_hardware, metrics_collector=self.metrics_collector, model_profile=model_profile_data)
         * self.llm_framework_instances.append(instance)
   5. Initialize Workload Generator:
   * Create target_frameworks_map = {fw.framework_id: fw for fw in self.llm_framework_instances if fw_config_entry.is_target_for_workload}.
   * More accurately, the workload generator might need the handle_incoming_request method or a simpy.Store from each target framework.
   * target_dispatch_points = {}
   * For fw_instance in self.llm_framework_instances:
Find the original config entry for this instance to check is_target_for_workload
original_fw_config = next(fc for fc in self.config.frameworks_to_test if fc.name == fw_instance.framework_id)
if original_fw_config.is_target_for_workload:
# Assuming frameworks have a 'request_arrival_queue' (simpy.Store)
# or a method to directly handle requests.
# Using the queue approach from AbstractLLMFramework:
target_dispatch_points[fw_instance.framework_id] = fw_instance.request_arrival_queue
   * self.workload_generator = WorkloadGenerator(simpy_env_instance, self.config.workload, target_dispatch_points, self.metrics_collector, self.virtual_hardware)
      * ExperimentOrchestrator.run(self):
      1. Calls self.setup_simulation().
      2. Log experiment start (e.g., using Python's logging module).
      3. Start LLM Framework Processing Loops:
      * For fw_instance in self.llm_framework_instances:
      * self.sim_env_wrapper.schedule_process(fw_instance.processing_loop)
      4. Start Workload Generator:
      * self.sim_env_wrapper.schedule_process(self.workload_generator.generate_requests_process)
      5. Run Core Simulation:
      * self.sim_env_wrapper.run() (This blocks until simulation ends).
      6. Generate and Save Reports:
      * simulation_actual_duration = self.sim_env_wrapper.now()
      * summary_report_dict = self.metrics_collector.generate_summary_report(simulation_actual_duration)
      * Print summary report to console (e.g., using pprint or custom formatting).
      * If self.config.metrics_config.output_summary_json_path:
      * Save summary_report_dict to JSON file.
      * If self.config.metrics_config.output_requests_csv_path and hasattr(self.metrics_collector, 'get_all_request_metrics_df'):
      * df_requests = self.metrics_collector.get_all_request_metrics_df()
      * Save df_requests to CSV file.
      7. Log experiment end.
      8. Return summary_report_dict.
4. Interfaces/APIs:
      * Constructor: ExperimentOrchestrator(config_data: dict)
      * run() -> dict: Main method to set up and run the entire experiment, returns the summary report.
5. Configuration Parameters:
      * The entire experiment configuration, typically loaded from a single YAML or JSON file, structured as described previously.
Example YAML Structure Snippet (refer to detailed one in previous thought process):
# experiment_config.yaml
simulation:
 max_simulation_time: 3600 # seconds
 random_seed: 42

model_characteristics_db_path: "./model_data/model_params.json" # Path relative to execution

hardware_profile:
 compute_devices:
   - device_id: "gpu0"
     device_type: "GPU"
     peak_tflops: {"fp16": 312, "int8": 624} # TFLOPs
     memory_capacity_bytes: 80_000_000_000 # 80GB
     memory_gbps: 2000 # GB/s
     processing_units: 108 # e.g., SM count
 network_links:
   - link_id: "client_to_fw_entry"
     source_id: "client_node_0"
     dest_id: "framework_entry_0" # Abstract entry point
     bandwidth_bps: 10_000_000_000 # 10 Gbps
     latency_s: 0.0001 # 100 microseconds

workload:
 total_duration: 300 # Shorter for example
 bytes_per_token_estimate_for_network: 2
 random_seed: 123
 client_profiles:
   - profile_name: "interactive_chat"
     weight: 1.0
     inter_arrival_time_dist_config: {"type": "Exponential", "rate": 10.0} # 10 reqs/sec avg
     prompt_tokens_dist_config: {"type": "LogNormal", "mean": 50, "sigma": 20, "is_int": True}
     max_output_tokens_dist_config: {"type": "Uniform", "low": 50, "high": 200, "is_int": True}
     conversational_probability: 0.3
     streaming_response_probability: 0.9

frameworks_to_test:
 - name: "vllm_Llama3_8B_fp16"
   type: "VLLM" # Maps to VLLMFramework class
   is_target_for_workload: true
   config:
     model_profile_id: "Llama3-8B" # Must exist in model_characteristics_db.json
     gpu_id: "gpu0"
     block_size: 16
     max_num_seqs: 256
     max_num_batched_tokens: 4096
     scheduler_iteration_delay_s: 0.0001 # Very small delay for responsiveness
     bytes_per_token_estimate_for_network: 2

metrics_config:
 percentiles_to_calculate: [0.5, 0.9, 0.95, 0.99]
 warm_up_duration_s: 60
 output_summary_json_path: "results/summary.json"
 output_requests_csv_path: "results/detailed_requests.csv"

6. Dependencies:
         * All other modules: CoreSimulationEngine, VirtualHardwarePlatform, WorkloadGenerator, AbstractLLMFramework (and its concrete implementations like VLLMFramework), MetricsCollector.
         * YAML/JSON parsing library (e.g., PyYAML, json).
         * (Optional) pandas for CSV output.
         * Python's logging module for console/file logging.
         * pprint for pretty printing reports.
         * Potentially Pydantic for config validation.