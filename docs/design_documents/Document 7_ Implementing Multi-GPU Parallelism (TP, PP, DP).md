Document 7: Implementing Multi-GPU Parallelism (TP, PP, DP)
1. Introduction
* Purpose: This document outlines the design for integrating multi-GPU parallelism strategies—Tensor Parallelism (TP), Pipeline Parallelism (PP), and Data Parallelism (DP)—into the virtual LLM serving testing environment. The goal is to enable simulation and comparison of frameworks leveraging these techniques for large model inference.
* Core Principle: The VirtualHardwarePlatform (Doc 2) provides the foundational multi-GPU and inter-GPU network primitives. The specific logic for implementing and managing a given parallelism strategy resides within the respective Pluggable LLM Framework Module (Doc 4).
* Scope: This document details necessary configuration changes, modifications to the abstract framework module, and the core logic for implementing TP, PP, and DP within concrete framework modules.
2. Configuration of Parallelism
The primary configuration for parallelism will reside within the frameworks_to_test section of the main experiment_config.yaml (Doc 6).
* Updates to frameworks_to_test[n].config:
Each framework instance will have a new parallelism sub-configuration object:
# In experiment_config.yaml
frameworks_to_test:
 - name: "MyParallelFramework_TP4_PP2"
   type: "CustomParallelFramework" # Maps to a specific framework class
   is_target_for_workload: true
   config:
     model_profile_id: "LargeModel-100B"
     # ... other framework-specific configs ...
     parallelism:
       strategy: "TP_PP" # Options: "None", "TP", "PP", "DP_Replicated", "TP_PP"
       tp_degree: 4    # Tensor Parallelism degree (number of GPUs per TP group)
       pp_stages: 2    # Pipeline Parallelism stages
       # dp_replicas: 1 # For DP_Replicated, this is implicitly 1 per instance.
                       # For a different DP type, this might be > 1.
       # List of virtual GPU IDs this framework instance will manage for its parallel execution.
       # The length of this list must be consistent with the strategy.
       # e.g., for TP_PP with tp_degree=4, pp_stages=2, total GPUs = 4*2 = 8.
       gpu_ids: ["gpu0", "gpu1", "gpu2", "gpu3", "gpu4", "gpu5", "gpu6", "gpu7"]
       # Mapping of PP stages to their constituent TP groups (list of GPU IDs per stage)
       # This would be derived by the framework module based on gpu_ids, tp_degree, and pp_stages.
       # Example derivation:
       # stage_0_gpus: ["gpu0", "gpu1", "gpu2", "gpu3"]
       # stage_1_gpus: ["gpu4", "gpu5", "gpu6", "gpu7"]

* Updates to model_characteristics_db.json (Doc 2, Section 5):
To effectively simulate sharded operations, the model profile needs more granularity.
// model_characteristics_db.json
{
 "LargeModel-100B": {
   "total_parameters": 100e9,
   "num_layers": 80,
   "hidden_size": 8192,
   "num_attention_heads": 64, // Total heads before TP
   "kv_cache_bytes_per_token_per_layer": "...", // As before
   "layers": [ // Optional: Per-layer details if needed for fine-grained PP stage assignment
     {
       "layer_id": 0,
       "type": "Attention", // Or "SelfAttention", "CrossAttention"
       "flops_per_token_prefill": 10e9, // FLOPs for this layer type
       "flops_per_token_decode": 12e9,
       "memory_bytes_per_token_prefill": 2e6,
       "memory_bytes_per_token_decode": 2.5e6,
       "activation_output_bytes_per_token": 8192 * 2, // hidden_size * bytes_per_element (e.g., FP16)
       "is_tp_shardable_qkv": true, // QKV projection
       "is_tp_shardable_output_proj": true, // Output projection
       "is_tp_collective_type": "AllReduce" // or "ReduceScatter_AllGather"
     },
     {
       "layer_id": 1,
       "type": "MLP",
       "flops_per_token_prefill": 20e9,
       // ... similar fields ...
       "is_tp_shardable_gate_up": true,
       "is_tp_shardable_down_proj": true,
       "is_tp_collective_type": "AllReduce"
     }
     // ... more layers
   ],
   // Fallback if per-layer details are not provided or needed
   "default_attention_stats": { /* ... */ },
   "default_mlp_stats": { /* ... */ }
 }
}

   * The framework module will use these details to calculate computation for sharded parts and data sizes for collectives.
3. Tensor Parallelism (TP) Implementation
   * Framework Module Logic (AbstractLLMFramework and concrete implementations):
   * Initialization (__init__):
   * Parse parallelism.tp_degree and parallelism.gpu_ids.
   * Validate len(gpu_ids) is a multiple of tp_degree if combined with PP, or equals tp_degree if TP only.
   * Create internal mapping for TP ranks (e.g., self.tp_rank_to_gpu_id).
   * _dispatch_tp_shardable_operation(self, base_op_stats, batch_size, sequence_length, op_type, tp_collective_type) (Helper Method):
   1. tp_degree = self.config.parallelism.tp_degree
   2. tp_gpu_group = self.current_tp_gpu_group (list of GPU IDs for the current TP execution, e.g., one PP stage's GPUs)
   3. Calculate Sharded Compute:
   * sharded_flops = base_op_stats['flops_per_token'] / tp_degree
   * sharded_memory = base_op_stats['memory_bytes_per_token'] / tp_degree (approximate, actual memory access patterns can be complex).
   * task_description_template = {'flops_required_fp16': sharded_flops * sequence_length * batch_size, ...}
   4. Submit Parallel Compute Tasks:
   * compute_events = []
   * For gpu_id in tp_gpu_group:
   * compute_events.append(self.simpy_env.process(self.virtual_hardware.submit_computation_task(gpu_id, task_description_template)))
   * yield simpy.AllOf(self.simpy_env, compute_events) (Wait for all shards to compute).
   5. Simulate Collective Communication:
   * Estimate data_size_for_collective (e.g., batch_size * sequence_length * hidden_size * bytes_per_element for activations).
   * This is a simplification. Actual collective performance depends on algorithm (e.g. ring, tree), message chunking, and number of links used. For simulation, we can model it as:
   * A series of point-to-point transfers if using a simple ring model.
   * Or, an abstract "collective cost" based on total data, tp_degree, and inter-GPU link characteristics. For simplicity, assume a cost proportional to data moved over the slowest link in the group, or an aggregate bandwidth.
   * A simplified approach: one submit_network_transfer_task per link involved in a conceptual ring, or a single delay.
   * Example for AllReduce: Each GPU sends and receives data_size_for_collective.
   * collective_events = []
   * For i from 0 to tp_degree-1: (Conceptual ring all-reduce)
   * source_gpu = tp_gpu_group[i]
   * dest_gpu = tp_gpu_group[(i + 1) % tp_degree]
   * # For AllReduce, data effectively travels 2*(N-1)/N * data_size per GPU over N steps
   * # Simplified: Assume data_size_for_collective / tp_degree is exchanged effectively
   * effective_transfer_size = data_size_for_collective / tp_degree # Highly abstract
   * collective_events.append(self.simpy_env.process(self.virtual_hardware.submit_network_transfer_task(source_gpu, dest_gpu, effective_transfer_size)))
   * yield simpy.AllOf(self.simpy_env, collective_events)
   * Integration into processing_loop (or per-layer methods):
When processing a model layer (e.g., Attention or MLP):
# Inside processing_loop or a layer-specific method
# if layer_info['is_tp_shardable_...']:
#     base_stats = self.model_profile['layers'][layer_idx] # or default_attention_stats
#     yield self.simpy_env.process(self._dispatch_tp_shardable_operation(
#         base_stats, current_batch_size, current_seq_len, 
#         op_type='AttentionQKV', 
#         tp_collective_type=layer_info.get('is_tp_collective_type', 'AllReduce')
#     ))
# else: # Non-TP operation (e.g., LayerNorm, or if TP degree is 1)
#     # Submit computation to the primary GPU of the current group/stage
#     primary_gpu = self.current_tp_gpu_group[0]
#     yield self.simpy_env.process(self.virtual_hardware.submit_computation_task(primary_gpu, ...))

4. Pipeline Parallelism (PP) Implementation
      * Framework Module Logic:
      * Initialization (__init__):
      * Parse parallelism.pp_stages and parallelism.gpu_ids.
      * self.num_layers_total = self.model_profile['num_layers']
      * self.layers_per_stage = self.num_layers_total // self.config.parallelism.pp_stages (can be more complex for uneven distribution).
      * self.stage_to_gpu_map = {} (maps stage index to a list of GPU IDs if TP is used per stage, or a single GPU ID if no TP per stage). This is derived from parallelism.gpu_ids, tp_degree, and pp_stages.
      * self.stage_layer_ranges = {} (maps stage index to (start_layer_idx, end_layer_idx)).
      * processing_loop(self) (Main changes here):
      * The loop will manage requests and their progression through pipeline stages as micro-batches.
      * A request might be broken into num_microbatches.
      * Maintain state for each micro-batch: (request_id, microbatch_idx, current_stage_idx, data_location_gpu_id, arrival_time_at_stage).
      * Use simpy.Store for input queues for each stage: self.stage_input_queues[stage_idx].
      * Create separate SimPy processes for each pipeline stage worker: _pipeline_stage_worker_process(self, stage_idx).
# In __init__ or setup method:
# self.stage_input_queues = [simpy.Store(self.simpy_env) for _ in range(self.config.parallelism.pp_stages)]
# for i in range(self.config.parallelism.pp_stages):
#     self.simpy_env.process(self._pipeline_stage_worker_process(i))

# Main request handling part of processing_loop (simplified):
# request = yield self.request_arrival_queue.get()
# num_microbatches = self.config.get("num_microbatches_per_request", 4)
# for mb_idx in range(num_microbatches):
#    microbatch_data = {"req_id": request.request_id, "mb_idx": mb_idx, "data_size_tokens": request.prompt_num_tokens / num_microbatches, ...}
#    yield self.stage_input_queues[0].put(microbatch_data)

      * _pipeline_stage_worker_process(self, stage_idx) (SimPy Process):
# def _pipeline_stage_worker_process(self, stage_idx):
#     my_gpu_ids_for_this_stage = self.stage_to_gpu_map[stage_idx]
#     # self.current_tp_gpu_group can be set here if TP is active for this stage
#     while True:
#         microbatch = yield self.stage_input_queues[stage_idx].get()
#         
#         # Perform computation for layers in this stage
#         start_layer, end_layer = self.stage_layer_ranges[stage_idx]
#         for layer_idx in range(start_layer, end_layer + 1):
#             layer_info = self.model_profile['layers'][layer_idx]
#             # If TP enabled for this stage, use _dispatch_tp_shardable_operation
#             # Else, compute on my_gpu_ids_for_this_stage[0]
#             # yield self.simpy_env.process(compute_for_layer(...))
#
#         # If not the last stage, transfer activations to the next stage
#         if stage_idx < self.config.parallelism.pp_stages - 1:
#             activation_size = microbatch['data_size_tokens'] * layer_info['activation_output_bytes_per_token']
#             source_gpu = my_gpu_ids_for_this_stage[-1] # Assuming last GPU of current stage group sends
#             dest_gpu_of_next_stage_input = self.stage_to_gpu_map[stage_idx + 1][0] # Assuming first GPU of next stage group receives
#
#             yield self.simpy_env.process(self.virtual_hardware.submit_network_transfer_task(
#                 source_gpu, dest_gpu_of_next_stage_input, activation_size
#             ))
#             # Put processed microbatch into the next stage's queue
#             yield self.stage_input_queues[stage_idx + 1].put(microbatch) # Update microbatch state
#         else:
#             # Last stage: microbatch processing complete for this request
#             # Aggregate microbatch results (conceptually) and log metrics
#             # self.metrics_collector.log_microbatch_completion(...)
#             # If all microbatches for a request are done, log request completion.

      * KV Cache Handling: Each stage is responsible for its portion of the KV cache. If a stage is TP sharded, the KV cache for its layers is also TP sharded across the GPUs of that stage. No KV cache transfer between PP stages is typically simulated unless the model architecture specifically requires it (rare for standard decoders).
5. Data Parallelism (DP) Implementation (for Inference Throughput)
         * Primary Method: DP via Replicated Instances (strategy: "DP_Replicated")
         * This is the most common and straightforward DP for inference.
         * Configuration: The user defines multiple framework instances in experiment_config.yaml, each potentially on a different gpu_id (or set of gpu_ids if each replica itself uses TP/PP).
frameworks_to_test:
 - name: "ModelReplica1_GPU0"
   type: "VLLM" # Or any other framework
   is_target_for_workload: true
   config:
     model_profile_id: "MyModel"
     gpu_id: "gpu0" # Single GPU for this replica
     # ...
 - name: "ModelReplica2_GPU1"
   type: "VLLM"
   is_target_for_workload: true
   config:
     model_profile_id: "MyModel"
     gpu_id: "gpu1" # Single GPU for this replica
     # ...

         * ExperimentOrchestrator (Doc 6): Already handles creating these multiple instances.
         * WorkloadGenerator (Doc 3): Needs a mechanism to distribute incoming requests among these target DP replicas (e.g., round-robin, least loaded based on queue depth if accessible).
            * The target_framework_queues or target_framework_handlers in WorkloadGenerator would contain all DP replicas marked as is_target_for_workload: true.
            * No change to Framework Module internal logic for DP itself: Each replica operates independently on its assigned GPU(s).
            * Alternative: Within-Framework DP (More Complex, Less Common for Latency-Critical Inference)
            * If a single framework instance needs to manage DP over a very large batch across its parallelism.gpu_ids.
            * The framework module's processing_loop would:
            1. Take a large incoming batch.
            2. Split the batch into sub-batches, one for each DP rank/GPU in parallelism.gpu_ids.
            3. Each GPU processes its sub-batch (requires model replication or appropriate sharding on each GPU). This is like running mini-replicas internally.
            4. No inter-GPU communication is needed during sub-batch processing itself (unlike TP/PP).
            5. Results from sub-batches are gathered.
6. Combined Parallelism (e.g., TP + PP, strategy: "TP_PP")
            * The framework module needs to orchestrate both.
            * Initialization:
            * gpu_ids are partitioned first for PP stages.
            * Then, within each PP stage, the assigned GPUs are used for TP.
            * Example: gpu_ids = [g0,g1,g2,g3], pp_stages=2, tp_degree=2.
            * Stage 0: GPUs [g0,g1], TP degree 2.
            * Stage 1: GPUs [g2,g3], TP degree 2.
            * _pipeline_stage_worker_process(self, stage_idx):
            * Before processing layers, it sets self.current_tp_gpu_group = self.stage_to_gpu_map[stage_idx].
            * When calling layer computation logic, it checks is_tp_shardable. If so, it calls _dispatch_tp_shardable_operation which then operates on self.current_tp_gpu_group.
            * Activation transfers between PP stages occur as before, from the designated GPU(s) of the source stage to the designated GPU(s) of the destination stage.
7. Impact on AbstractLLMFramework
            * The __init__ signature will now formally expect parallelism_config as part of framework_specific_config.
            * The base class might provide utility methods for parsing parallelism_config or common calculations if they are truly generic (though much logic will be framework-specific).
            * No new abstract methods are strictly required, as the processing_loop is the main extension point. However, concrete implementations will become substantially more complex.
8. Key Considerations & Simplifications
            * Collectives Modeling: Accurately modeling the performance of collective communications (all-reduce, all-gather, etc.) is complex. The initial implementation can use simplified models (e.g., based on total data transferred over a conceptual link or sum of point-to-point transfers in a ring). More advanced models (e.g., NCCL cost models) could be integrated later.
            * Synchronization Overheads: SimPy's yield and AllOf naturally handle synchronization delays. Additional small, constant overheads for synchronization primitives can be added if deemed significant.
            * Memory Allocation for Parallel Ops: The VirtualHardwarePlatform.allocate_memory needs to be called for the correct GPU when memory for sharded weights, activations, or KV cache is allocated.
            * Model Slicing Logic: The framework module is responsible for how the model (layers, attention heads, MLP dimensions) is sliced/sharded based on TP/PP configuration. The model_characteristics_db provides the raw data; the module interprets it.
This document provides a design for implementing TP, PP, and DP. The complexity lies primarily within the concrete framework modules, which must accurately translate their respective parallelism strategies into sequences of computation and network tasks for the VirtualHardwarePlatform.