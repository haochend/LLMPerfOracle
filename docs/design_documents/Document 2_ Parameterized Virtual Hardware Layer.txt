Document 2: Parameterized Virtual Hardware Layer
1. Purpose:
To model the performance characteristics of various hardware components (GPUs, CPUs, Memory, Network links) in an abstract, parameterized way. It provides an interface for other modules to request hardware resources and get estimated processing/transfer times based on configured capabilities and current contention.
2. Key Data Structures/Classes:
* HardwareResourceProfile (Data Class):
   * Purpose: Holds the configuration for all hardware components. Likely loaded from a YAML/JSON file.
   * Attributes:
      * compute_devices: List[Dict]: Configuration for each compute device.
      * memory_devices: List[Dict]: Configuration for distinct memory devices (if any).
      * network_links: List[Dict]: Configuration for network links.
* VirtualComputeDevice Class:
   * Purpose: Represents a single compute device (e.g., a GPU or CPU).
   * Attributes:
      * device_id: str
      * device_type: str (e.g., "GPU", "CPU")
      * simpy_env: simpy.Environment
      * processing_units_resource: simpy.Resource: Models parallel execution units (e.g., SMs for a GPU, cores for a CPU). Capacity set from config.
      * peak_tflops: Dict[str, float]: e.g., {"fp16": 312, "int8": 624}.
      * memory_capacity_bytes: int
      * memory_gbps: float: Memory bandwidth.
      * memory_available_container: simpy.Container: Models available memory capacity. init and capacity set to memory_capacity_bytes.
      * memory_bandwidth_resource: simpy.Resource: Models memory bandwidth contention. capacity=1 (conceptual, actual time derived from size/bandwidth_gbps).
      * model_characteristics_db: Dict: Reference to a database/dict of model layer properties (FLOPs, bytes).
* VirtualNetworkLink Class:
   * Purpose: Represents a network link between two points or nodes.
   * Attributes:
      * link_id: str
      * source_id: str
      * dest_id: str
      * simpy_env: simpy.Environment
      * bandwidth_bps: float: Bandwidth in bits per second.
      * latency_s: float: Latency in seconds.
      * link_utilization_resource: simpy.Resource(capacity=1): To model that the link is busy during transfer and contention.
* VirtualHardwarePlatform Class:
   * Purpose: Manages all virtual hardware devices and links. Provides the main API for resource requests.
   * Attributes:
      * simpy_env: simpy.Environment
      * compute_devices: Dict[str, VirtualComputeDevice]
      * network_links: Dict[Tuple[str, str], VirtualNetworkLink] (key could be (source_id, dest_id))
      * model_characteristics_db: Dict (e.g., loaded from JSON)
3. Core Logic/Algorithms:
* VirtualHardwarePlatform.initialize(profile: HardwareResourceProfile, simpy_env: simpy.Environment, model_chars_db: Dict):
   1. Stores simpy_env and model_chars_db.
   2. For each compute device config in profile.compute_devices:
      * Creates a VirtualComputeDevice instance.
      * Initializes its SimPy resources (processing_units_resource with specified capacity, memory_available_container, memory_bandwidth_resource).
      * Stores it in self.compute_devices.
   3. For each network link config in profile.network_links:
      * Creates a VirtualNetworkLink instance.
      * Initializes its SimPy resource (link_utilization_resource).
      * Stores it in self.network_links (ensure bidirectional access or define links one-way and expect caller to find reverse if needed).
* VirtualHardwarePlatform.submit_computation_task(self, device_id: str, task_description: Dict) -> simpy.Process:
   * task_description keys: task_id, flops_required_fp16 (or other precision), memory_read_bytes, memory_write_bytes, num_processing_units_requested (e.g., 1 SM for a small op, or all for a large matmul), is_memory_bound_hint (optional, from Roofline pre-analysis).
   * Logic (as a SimPy process def _computation_task_process(...)):
      1. Get device = self.compute_devices[device_id].
      2. yield device.processing_units_resource.request() (requesting 1 unit of the resource for now; can be num_processing_units_requested).
      3. yield device.memory_bandwidth_resource.request() (conceptual, to serialize memory access).
      4. Calculate compute_time_s: task_description['flops_required_fp16'] / (device.peak_tflops['fp16'] * 1e12). (Adjust for actual units requested vs. total, and efficiency factor if any).
      5. Calculate memory_access_time_s: (task_description['memory_read_bytes'] + task_description['memory_write_bytes']) / (device.memory_gbps * 1e9).
      6. Determine effective_time_s:
         * If is_memory_bound_hint is true, effective_time_s = memory_access_time_s.
         * Else (compute-bound), effective_time_s = compute_time_s.
         * (Simplified Roofline: max(compute_time_s, memory_access_time_s) or a more complex model if interactions are considered).
      7. yield self.simpy_env.timeout(effective_time_s).
      8. device.memory_bandwidth_resource.release().
      9. device.processing_units_resource.release().
      10. Log task completion (via Metrics Collector).
* VirtualHardwarePlatform.allocate_memory(self, device_id: str, size_bytes: int) -> simpy.Process:
   * Logic (as a SimPy process def _allocate_memory_process(...)):
      1. Get device = self.compute_devices[device_id].
      2. yield device.memory_available_container.get(size_bytes). (SimPy handles queuing if not enough space).
      3. Log memory allocation.
* VirtualHardwarePlatform.free_memory(self, device_id: str, size_bytes: int) -> simpy.Process:
   * Logic (as a SimPy process def _free_memory_process(...)):
      1. Get device = self.compute_devices[device_id].
      2. yield device.memory_available_container.put(size_bytes).
      3. Log memory deallocation.
* VirtualHardwarePlatform.submit_network_transfer_task(self, source_node_id: str, dest_node_id: str, data_size_bytes: int) -> simpy.Process:
   * Logic (as a SimPy process def _network_transfer_process(...)):
      1. Find link = self.network_links.get((source_node_id, dest_node_id)) (or handle lookup for appropriate link). If not found, error or use default.
      2. yield link.link_utilization_resource.request().
      3. yield self.simpy_env.timeout(link.latency_s).
      4. transfer_time_s = (data_size_bytes * 8) / link.bandwidth_bps. (Convert bytes to bits). Add protocol_overhead_factor if any.
      5. yield self.simpy_env.timeout(transfer_time_s).
      6. link.link_utilization_resource.release().
      7. Log network transfer completion.
4. Interfaces/APIs (for LLM Framework Modules, Workload Generator):
* initialize(profile, simpy_env, model_chars_db)
* submit_computation_task(device_id, task_description): Returns a SimPy process event that can be yielded on.
* allocate_memory(device_id, size_bytes): Returns a SimPy process event.
* free_memory(device_id, size_bytes): Returns a SimPy process event.
* submit_network_transfer_task(source_node_id, dest_node_id, data_size_bytes): Returns a SimPy process event.
* get_device_info(device_id) -> VirtualComputeDevice: To query device static parameters if needed.
5. Configuration Parameters:
* A detailed hardware profile (YAML/JSON) specifying all compute devices (GPUs, CPUs with their TFLOPs, memory capacity/bandwidth, processing units) and network links (source, destination, latency, bandwidth).
* Path to a "Model Characteristics Database" (JSON/dict):
// model_characteristics_db.json
{
 "Llama2-7B": {
   "parameters": 7e9,
   "hidden_size": 4096,
   // For prefill phase (per token, per layer, or aggregated)
   "prefill_op_stats": {"flops_per_token": 2e9, "memory_bytes_per_token": 1e3}, // Simplified example
   // For decode phase (per token, per layer, or aggregated)
   "decode_op_stats": {"flops_per_token": 3e9, "memory_bytes_per_token": 1.5e3}, // Simplified example
   "kv_cache_bytes_per_token_per_layer": 2 * 4096 * 2, // Assuming FP16 for K and V
   "num_layers": 32
 },
 // ... other models
}

(The task_description passed to submit_computation_task would be derived by the LLM Framework module from these model characteristics, the current request's sequence length, batch size etc.)
6. Dependencies:
   * Core Simulation Engine (for simpy_env).
   * Metrics Collector (for logging hardware utilization, task completions).