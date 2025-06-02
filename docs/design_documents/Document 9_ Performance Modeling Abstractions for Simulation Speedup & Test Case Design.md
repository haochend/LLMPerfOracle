Execution Document 9: Performance Modeling Abstractions for Simulation Speedup & Test Case Design
1. Introduction
* Purpose: To outline strategies for abstracting low-level computational and communication details (like per-layer execution and fine-grained collective communication steps) into higher-level analytical models. This will significantly reduce the number of discrete events and Python overhead, speeding up the simulation while aiming to preserve acceptable accuracy for overall system performance metrics. This document also includes test case designs to demonstrate and quantify the achieved speedup.
* Core Principle: Replace sequences of fine-grained SimPy events with pre-calculated or analytically derived single-duration events for larger operational blocks.
2. Strategy 1: Macro Operations for Intra-Model Computation (Prefill/Decode)
This strategy addresses the slowdown from simulating "multi-layers."
* Current (Potentially Slow) Design Implication: If an LLMFrameworkModule (Doc 4 & 7) simulates prefill or decode by iterating through each layer defined in model_characteristics_db.json and submitting a separate computation_task to the VirtualHardwarePlatform (Doc 2) for each layer, this generates many events.
* Refinement - Aggregated Operational Profiles:
   1. Enhanced model_characteristics_db.json:
Instead of (or in addition to) per-layer stats, provide aggregated statistics for the entire prefill operation for a given number of prompt tokens, and for a single decode step for a given batch size.
// model_characteristics_db.json
{
 "Llama3-8B": {
   // ... existing model parameters ...
   "aggregated_ops": {
     "prefill": { // Characterized for a representative number of prompt tokens, e.g., 512
       "total_flops_per_prompt_token": 1.5e9, 
       "total_memory_bytes_per_prompt_token": 1e6,
       "critical_path_factor": 1.0 
     },
     "decode": { // Characterized for a single token generation across a batch
       "total_flops_per_token_in_batch": 2e9, 
       "total_memory_bytes_per_token_in_batch": 1.2e6,
       "critical_path_factor": 1.0
     }
   },
   "layers": [ /* ... per-layer stats can still exist for PP or very detailed LoD ... */ ]
 }
}

   2. Modified LLMFrameworkModule Logic:
   * The _estimate_prefill_ops method would now calculate total FLOPs and memory for the entire prefill phase based on current_sequence_state.num_tokens_requiring_prefill and the new aggregated stats:
# In LLMFrameworkModule
def _estimate_macro_prefill_ops(self, model_profile, num_tokens_to_prefill):
   agg_stats = model_profile['aggregated_ops']['prefill']
   total_flops = agg_stats['total_flops_per_prompt_token'] * num_tokens_to_prefill
   total_memory_bytes = agg_stats['total_memory_bytes_per_prompt_token'] * num_tokens_to_prefill
   return {'flops_required_fp16': total_flops, 'memory_read_bytes': total_memory_bytes, 'memory_write_bytes': total_memory_bytes / 2} # Example R/W split

   * Similarly, _estimate_macro_decode_op(self, model_profile, current_batch_size) for the decode step.
   * The framework's processing_loop then calls virtual_hardware.submit_computation_task once for the entire prefill of a sequence (or batch of sequences if prefills are batched) and once per iteration for the batched decode step.
      3. Impact: Reduces dozens or hundreds of per-layer simulation events for each prefill/decode into a single computation event.
      4. Pipeline Parallelism (PP) Consideration: If PP is used, these aggregated ops would be for the set of layers within a single PP stage. The framework module for PP would need to sum up the total_flops_per_prompt_token (etc.) for layers belonging to that stage from the per-layer stats if aggregated_ops are not provided per-stage.
3. Strategy 2: Analytical Models for Collective Communications (TP)
This strategy addresses the slowdown from simulating "collective communications."
      * Current (Potentially Slow) Design Implication: _dispatch_tp_shardable_operation (Doc 7) might simulate collectives like AllReduce by creating multiple submit_network_transfer_task SimPy processes in a loop.
      * Refinement - Collective Cost Models:
      1. Implement Analytical Cost Functions:
Create helper functions to estimate the time for common collective operations.
# Example: simplified_ring_allreduce_time(data_size_bytes, num_gpus, link_bandwidth_bps, link_latency_s)
def simplified_ring_allreduce_time(data_size_bytes, num_gpus, link_bandwidth_bps, link_latency_s):
   if num_gpus <= 1:
       return 0.0
   # Alpha-Beta model: latency term (alpha) + bandwidth term (beta * size)
   # For ring all-reduce, roughly: 2 * (N-1)/N * (data_size / bandwidth) + 2 * (N-1) * latency
   # This is a simplification. More accurate models like those from NCCL docs can be used.
   # time_bw = (data_size_bytes * 8 / link_bandwidth_bps) * 2 * (num_gpus -1) / num_gpus 
   time_bw = (data_size_bytes * 8 / link_bandwidth_bps) # Time to send all data once
   # In a ring, each piece of data traverses N-1 links.
   # For all-reduce, data is reduced and broadcast. Total data movement is complex.
   # A common model: alpha * 2*(N-1) + beta * data_size * 2*(N-1)/N
   # Let's use a simpler version for illustration:
   communication_time_per_step = link_latency_s + (data_size_bytes * 8 / link_bandwidth_bps)
   total_collective_time = 2 * (num_gpus - 1) * communication_time_per_step # Highly abstract, assumes full data per step
   return total_collective_time 

Similar models for AllGather, ReduceScatter, etc. These models would take parameters of the specific inter-GPU links from the VirtualHardwarePlatform.
      2. Modified _dispatch_tp_shardable_operation:
         * After parallel compute shards complete:
         1. Estimate data_size_for_collective.
         2. Get relevant link_bandwidth_bps and link_latency_s for the TP group's interconnect.
         3. collective_time_s = simplified_ring_allreduce_time(...).
         4. yield self.simpy_env.timeout(collective_time_s).
         5. (Optional Contention Modeling):
# links_used = get_links_for_tp_group(...)
# link_requests = [link.link_utilization_resource.request() for link in links_used]
# yield simpy.AllOf(self.simpy_env, link_requests) 
# yield self.simpy_env.timeout(collective_time_s)
# for req_tuple in link_requests: # Assuming request returns a tuple (request_event, link_resource)
#    link_resource = req_tuple[1] # Or however the resource is accessed
#    link_resource.release(req_tuple[0])

            3. Impact: Reduces many network events per collective into one timeout event.
4. Configurable Level of Detail (LoD)
            * Introduce an LoD Parameter: In experiment_config.yaml under simulation or per framework.
            * lod: "high" (detailed simulation: per-layer, per-packet/step collective)
            * lod: "medium" (abstracted: macro-ops for layers, analytical collectives)
            * Conditional Logic: Framework modules use this lod to switch simulation logic.
# In LLMFrameworkModule
# if self.config.simulation_lod == "high":
#    # Detailed logic
# else: # "medium"
#    # Abstracted logic

5. Trade-offs
               * Speed: "Medium" LoD will be significantly faster.
               * Fidelity: Accuracy depends on the quality of aggregated profiles and analytical collective models. Per-layer contention or nuanced collective behaviors are abstracted.
               * Complexity: Requires upfront effort to develop and validate abstracted models.
6. Implementation Steps:
               1. Profile Existing Simulation: Identify event count bottlenecks.
               2. Develop Aggregated Profiles: For key models, derive aggregated_ops.
               3. Implement Analytical Collective Models: Research and implement cost functions.
               4. Refactor Framework Modules: Add lod switch and abstracted logic.
               5. Validate: Compare "medium" LoD metrics against "high" LoD (and real data if possible) to quantify accuracy trade-offs.
7. Test Case Design for Demonstrating Simulation Speedup
               * 7.1. Objectives of Speedup Testing:
               * Quantify the reduction in wall-clock simulation execution time when using "medium" LoD (abstracted models) compared to "high" LoD (detailed models).
               * Verify that key system-level performance metrics (e.g., throughput, average TTFT, average E2E latency) remain reasonably consistent or within an acceptable margin of error between LoD levels for the same logical scenario.
               * Identify which abstractions (macro-operations for layers, analytical collectives) contribute most to the speedup.
               * 7.2. Test Environment Setup:
               * Hardware for Running Simulation: A consistent machine to run the Python simulation itself.
               * Simulator Version: Use the same version of the simulation framework for all tests.
               * Python Profiler: Use a profiler like cProfile or line_profiler to measure execution time of different parts of the simulation code and count SimPy events.
               * Experiment Configuration: Prepare baseline experiment configuration files.
               * 7.3. Test Scenarios:
For each scenario, run the simulation twice: once with lod: "high" and once with lod: "medium".
                  * Simulated Workload: Keep the workload (request arrival pattern, prompt/output lengths) identical for both LoD runs within a scenario. Use a fixed random seed for the workload generator.
                  * Simulated Hardware: Keep the virtual hardware profile identical.
                  * Simulated Duration: Run for a fixed, meaningful simulated duration (e.g., 1 hour of simulated time) or a fixed number of total requests.
                  * 7.3.1. Scenario A: Single Large Model, Single GPU (Focus on Macro Operations for Layers)
                  * Model: A large LLM with many layers (e.g., 80 layers).
                  * Hardware: Single powerful virtual GPU.
                  * Parallelism: strategy: "None".
                  * Framework: A framework module that, in "high" LoD, iterates through all layers for prefill/decode.
                  * Hypothesis: "Medium" LoD (using aggregated ops for the entire prefill/decode pass) will be significantly faster due to fewer computation events.
                  * 7.3.2. Scenario B: Tensor Parallelism (Focus on Analytical Collectives)
                  * Model: A model suitable for TP.
                  * Hardware: Multiple virtual GPUs (e.g., 4 GPUs) with defined inter-GPU links.
                  * Parallelism: strategy: "TP", tp_degree: 4.
                  * Framework: A framework module implementing TP. In "high" LoD, it simulates collectives with multiple network events. In "medium" LoD, it uses a single analytical cost function and timeout.
                  * Hypothesis: "Medium" LoD will be faster due to fewer network events for collectives, especially with frequent decode steps.
                  * 7.3.3. Scenario C: Pipeline Parallelism with TP (Combined Focus)
                  * Model: Very large model.
                  * Hardware: Sufficient virtual GPUs for PP and TP (e.g., 8 GPUs).
                  * Parallelism: strategy: "TP_PP", pp_stages: 2, tp_degree: 4.
                  * Framework: A framework module implementing TP and PP.
                  * Hypothesis: "Medium" LoD will show the most significant speedup due to the combined effect of abstracting both per-layer computations within stages and inter-GPU collectives for TP.
                  * 7.4. Metrics to Measure (for each LoD run within a scenario):
                  1. Wall-Clock Simulation Time: The actual time taken to run the Python simulation script from start to finish. This is the primary indicator of speedup.
                  2. Total SimPy Events Processed: (If easily obtainable from SimPy or by instrumenting the event scheduler wrapper). A lower count indicates less simulation overhead.
                  3. Key Performance Metrics (from MetricsCollector):
                  * Average Time To First Token (TTFT)
                  * Average Time Per Output Token (TPOT)
                  * Average End-to-End Request Latency
                  * Overall Throughput (requests/sec and tokens/sec)
                  * Number of completed requests.
                  4. (Optional) Profiler Output: Breakdown of time spent in different functions/modules.
                  * 7.5. Expected Outcomes and Analysis:
                  * Speedup Factor: Calculate WallClockTime_HighLoD / WallClockTime_MediumLoD. Expect this to be significantly > 1.
                  * Metric Consistency: Compare the key performance metrics (TTFT, throughput, etc.) between "high" and "medium" LoD. Document the percentage difference. The goal is for these differences to be small enough that the "medium" LoD is still useful for system-level decision-making.
                  * Event Reduction: Report the reduction in total SimPy events.
                  * Bottleneck Shift: Profiler output might show that with abstractions, the simulation bottleneck shifts from SimPy event processing or Python loop overhead to other parts of the code (e.g., data management, metrics calculation if not optimized).
                  * Conclusion: Based on the speedup factor and metric consistency, determine the effectiveness of the abstractions and the suitability of the "medium" LoD for different types of studies.
By executing these test cases, the benefits of the performance modeling abstractions can be clearly demonstrated and quantified, providing confidence in using the faster "medium" LoD for more extensive simulation studies.