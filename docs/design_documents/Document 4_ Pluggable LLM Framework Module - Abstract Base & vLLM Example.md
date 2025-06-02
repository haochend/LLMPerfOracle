Document 4: Pluggable LLM Framework Module - Abstract Base & vLLM Example
1. Purpose:
To provide a standardized way to model the behavior and performance of different LLM serving frameworks, allowing them to be "plugged into" the simulation environment. This document details an abstract base class and a concrete implementation for vLLM as an example.
Part A: AbstractLLMFramework (Abstract Base Class)
A.1. Purpose:
Defines the common interface and essential lifecycle methods that all concrete LLM framework simulation modules must implement.
A.2. Key Abstract Methods (to be implemented by subclasses):
* __init__(self, framework_id: str, simpy_env: simpy.Environment, framework_specific_config: dict, virtual_hardware: VirtualHardwarePlatform, metrics_collector: MetricsCollector, model_profile: dict):
   * Stores common attributes like framework_id, simpy_env, config, virtual_hardware, metrics_collector, model_profile.
   * Initializes internal queues (e.g., self.request_arrival_queue = simpy.Store(simpy_env)).
   * Initializes any framework-specific state variables.
* handle_incoming_request(self, request: Request) -> simpy.Process:
   * Purpose: SimPy process initiated by the Workload Generator when a new request arrives for this framework.
   * Logic:
      1. Log receipt of the request.
      2. Perform any initial validation or queuing specific to the framework's entry point.
      3. yield self.request_arrival_queue.put(request).
* processing_loop(self) -> simpy.Process:
   * Purpose: The main SimPy process that embodies the framework's core logic: request scheduling, batch formation, KV cache management, interaction with virtual hardware for prefill/decode, and response generation.
   * Logic: This is highly framework-specific and will be detailed in concrete implementations. It typically involves a loop that:
      * Waits for new requests or for running tasks to complete.
      * Makes scheduling decisions.
      * Manages resources (e.g., KV cache).
      * Initiates prefill and decode operations via the VirtualHardwarePlatform.
      * Handles request completion and metrics logging.
* get_status(self) -> dict:
   * Purpose: Returns a snapshot of the framework's internal state for monitoring or debugging.
   * Example: {'queue_length': len(self.request_arrival_queue.items), 'active_sequences': len(self.active_sequences)}.
A.3. Common Attributes (available to subclasses):
* framework_id: str
* simpy_env: simpy.Environment
* config: dict (framework-specific part of the experiment config)
* virtual_hardware: VirtualHardwarePlatform
* metrics_collector: MetricsCollector
* model_profile: dict (characteristics of the LLM model being served, e.g., from model_characteristics_db.json)
* request_arrival_queue: simpy.Store
A.4. Utility Methods (optional, can be part of the base class if widely applicable):
* _estimate_prefill_ops(self, num_prompt_tokens: int, batch_size: int) -> Dict: Returns {flops, memory_read_bytes, memory_write_bytes} based on self.model_profile.
* _estimate_decode_op(self, batch_size: int) -> Dict: Returns {flops, memory_read_bytes, memory_write_bytes} for one decode step for the batch.
* _estimate_kv_cache_request_bytes(self, num_tokens: int) -> int: Returns estimated KV cache size based on self.model_profile.
Part B: VLLMFramework(AbstractLLMFramework) (Concrete Implementation Example)
B.1. Purpose:
Models the vLLM serving framework, focusing on its PagedAttention KV cache management and continuous batching scheduler.
B.2. Specific vLLM Mechanisms to Model:
* PagedAttention KV Cache:
   * Internal attributes:
      * gpu_id: str (from config, which virtual GPU to use)
      * block_size: int (number of tokens per KV cache block, from config)
      * num_gpu_blocks: int (total blocks available on the assigned GPU, derived from GPU memory and self.model_profile['kv_cache_bytes_per_token_per_layer'] and block_size)
      * gpu_blocks_container: simpy.Container (initialized with num_gpu_blocks capacity to manage free blocks)
   * Logic:
      * When a sequence is considered for prefill: Calculate required_blocks_for_prompt. Request these from gpu_blocks_container.
      * During decode: For each new token, calculate if a new block is needed. Request it.
      * On sequence completion/eviction: Release its blocks back to gpu_blocks_container.
* Continuous Batching Scheduler:
   * Internal attributes:
      * waiting_sequences: List[Request] (requests that have arrived but not yet started prefill)
      * running_sequences: Dict[str, SequenceState] (request_id -> state object for sequences currently in prefill or decode)
      * swapped_sequences: List[Request] (if preemption to CPU is modeled, optional for first pass)
      * max_running_sequences: int (from config, e.g., max_num_seqs)
      * max_batched_tokens_per_iteration: int (from config, e.g., max_num_batched_tokens)
   * SequenceState (Data Class): request_id, prompt_tokens_processed, output_tokens_generated, allocated_kv_blocks: List, status ('PREFILLING', 'DECODING', 'COMPLETED', 'WAITING_FOR_RESOURCES').
B.3. Core Logic for VLLMFramework.processing_loop(self) -> simpy.Process:
# Pseudocode for VLLMFramework.processing_loop
# This is a SimPy process, so 'yield' is used for time-passing operations.

def processing_loop(self):
   while True:
       # --- Part 1: Try to admit new sequences from arrival queue ---
       newly_admitted_requests_this_iteration = []
       if len(self.running_sequences) < self.config.get('max_num_seqs', float('inf')):
           requests_to_consider = []
           while len(self.request_arrival_queue.items) > 0:
                if self.request_arrival_queue.items: # if items are present
                   request = self.request_arrival_queue.items[0] # Peek
                   
                   required_blocks = self._calculate_prompt_kv_blocks(request.prompt_num_tokens)
                   if self.gpu_blocks_container.level >= required_blocks:
                       yield self.request_arrival_queue.get() # Consume it
                       yield self.simpy_env.process(
                           self.virtual_hardware.allocate_memory(self.gpu_id, required_blocks * self._bytes_per_kv_block())
                       ) 
                       yield self.gpu_blocks_container.get(required_blocks)

                       seq_state = SequenceState(request_id=request.request_id, request=request, status='WAITING_FOR_PREFILL', allocated_kv_blocks=[]) # Added request to SequenceState
                       self.running_sequences[request.request_id] = seq_state
                       self.waiting_sequences.append(request) 
                       newly_admitted_requests_this_iteration.append(request)
                   else:
                       break 
                else:
                   break

       # --- Part 2: Scheduling iteration (Prefill + Decode) ---
       prefill_batch = [] 
       decode_batch = [] 
       current_batched_tokens = 0

       for req_id, seq_state in list(self.running_sequences.items()): 
           if seq_state.status == 'DECODING' and current_batched_tokens < self.config.get('max_batched_tokens_per_iteration', float('inf')):
               decode_batch.append(seq_state)
               current_batched_tokens += 1 

       for request_obj in list(self.waiting_sequences): 
           if request_obj.request_id in self.running_sequences: 
               seq_state = self.running_sequences[request_obj.request_id]
               if seq_state.status == 'WAITING_FOR_PREFILL':
                   if (current_batched_tokens + request_obj.prompt_num_tokens) <= self.config.get('max_batched_tokens_per_iteration', float('inf')):
                       prefill_batch.append(seq_state)
                       current_batched_tokens += request_obj.prompt_num_tokens
                       self.waiting_sequences.remove(request_obj) 
                       seq_state.status = 'PREFILLING'
                   else:
                       break 

       # --- Part 3: Execute Prefill Batch ---
       if prefill_batch:
           prefill_processes = []
           for seq_state in prefill_batch:
               task_desc = self._estimate_prefill_ops(
                   self.model_profile,
                   seq_state.request.prompt_num_tokens,
                   batch_size=1 
               )
               task_desc['task_id'] = f"{seq_state.request.request_id}_prefill"
               
               self.metrics_collector.log_prefill_start(seq_state.request.request_id, self.simpy_env.now)
               seq_state.prefill_start_time = self.simpy_env.now
               
               prefill_proc = self.simpy_env.process(
                   self.virtual_hardware.submit_computation_task(self.gpu_id, task_desc)
               )
               prefill_processes.append(prefill_proc)

           yield simpy.AllOf(self.simpy_env, prefill_processes) 

           for seq_state in prefill_batch:
               if seq_state.request.request_id in self.running_sequences: 
                   seq_state.status = 'DECODING' 
                   seq_state.prompt_tokens_processed = seq_state.request.prompt_num_tokens
                   if seq_state not in decode_batch and current_batched_tokens < self.config.get('max_batched_tokens_per_iteration', float('inf')):
                        decode_batch.append(seq_state)
                        current_batched_tokens +=1

       # --- Part 4: Execute Decode Batch ---
       if decode_batch:
           task_desc_decode = self._estimate_decode_op(self.model_profile, batch_size=len(decode_batch))
           task_desc_decode['task_id'] = f"decode_batch_{self.simpy_env.now}"

           yield self.simpy_env.process(
               self.virtual_hardware.submit_computation_task(self.gpu_id, task_desc_decode)
           )
           
           current_time = self.simpy_env.now
           completed_in_this_decode_step = []

           for seq_state in decode_batch:
               if seq_state.request.request_id not in self.running_sequences: continue 

               if seq_state.output_tokens_generated == 0:
                   self.metrics_collector.log_first_token_generated(
                       seq_state.request.request_id,
                       current_time,
                       getattr(seq_state, 'prefill_start_time', current_time) # Use current_time if prefill_start_time not set
                   )

               seq_state.output_tokens_generated += 1
               self.metrics_collector.log_token_decoded(seq_state.request.request_id, current_time, seq_state.output_tokens_generated)

               token_data_size_bytes = 1 * self.config.get('bytes_per_token_estimate_for_network', 2)
               self.simpy_env.process(
                   self.virtual_hardware.submit_network_transfer_task(
                       self.gpu_id, 
                       "client_node_0", 
                       token_data_size_bytes
                   )
               )

               if seq_state.output_tokens_generated >= seq_state.request.max_output_tokens or \
                  self._check_eos_condition(seq_state): 
                   
                   self.metrics_collector.log_request_completed(
                       seq_state.request.request_id,
                       current_time,
                       seq_state.output_tokens_generated,
                       "SUCCESS"
                   )
                   completed_in_this_decode_step.append(seq_state.request.request_id)
               else:
                   if (seq_state.prompt_tokens_processed + seq_state.output_tokens_generated) % self.block_size == 0 :
                       can_allocate_block_event = self.gpu_blocks_container.get(1)
                       # Ensure simpy_env is accessible, e.g. self.simpy_env
                       block_allocated_result = yield can_allocate_block_event | self.simpy_env.timeout(0) 

                       if can_allocate_block_event not in block_allocated_result:
                           self.metrics_collector.log_request_completed( # Changed from log_request_completion_status
                               seq_state.request.request_id, current_time, seq_state.output_tokens_generated, "OOM_KV_BLOCK"
                           )
                           completed_in_this_decode_step.append(seq_state.request.request_id) 
                           yield self.simpy_env.process(self._release_sequence_resources(seq_state))

           for req_id in completed_in_this_decode_step:
               if req_id in self.running_sequences:
                   seq_to_remove = self.running_sequences.pop(req_id)
                   yield self.simpy_env.process(self._release_sequence_resources(seq_to_remove))

       # --- Part 5: Yield for a short duration or wait for an event ---
       if not prefill_batch and not decode_batch and not newly_admitted_requests_this_iteration:
           yield self.simpy_env.timeout(self.config.get('scheduler_iteration_delay_s', 0.001)) 

B.4. Helper methods for VLLMFramework:
* _calculate_prompt_kv_blocks(self, num_prompt_tokens: int) -> int: Estimates KV blocks for a prompt.
* _bytes_per_kv_block(self) -> int: self.block_size * self.model_profile['kv_cache_bytes_per_token_per_layer'] * self.model_profile['num_layers'].
* _release_sequence_resources(self, seq_state: SequenceState) -> simpy.Process: Releases KV blocks and other resources.
   * Logic: yield self.gpu_blocks_container.put(len(seq_state.allocated_kv_blocks))
   * yield self.virtual_hardware.free_memory(self.gpu_id, len(seq_state.allocated_kv_blocks) * self._bytes_per_kv_block())
* _check_eos_condition(self, seq_state: SequenceState) -> bool: Placeholder, always returns False for now.
* _estimate_prefill_ops(self, model_profile: dict, num_prompt_tokens: int, batch_size: int) -> dict:
   * Uses model_profile['prefill_op_stats'].
   * Returns {'flops_required_fp16': model_profile['prefill_op_stats']['flops_per_token'] * num_prompt_tokens * batch_size, 'memory_read_bytes': model_profile['prefill_op_stats']['memory_bytes_per_token'] * num_prompt_tokens * batch_size, 'memory_write_bytes': model_profile['prefill_op_stats']['memory_bytes_per_token'] * num_prompt_tokens * batch_size / 2}. (Example, needs refinement).
* _estimate_decode_op(self, model_profile: dict, batch_size: int) -> dict:
   * Uses model_profile['decode_op_stats'].
   * Returns {'flops_required_fp16': model_profile['decode_op_stats']['flops_per_token'] * batch_size, 'memory_read_bytes': model_profile['decode_op_stats']['memory_bytes_per_token'] * batch_size, 'memory_write_bytes': model_profile['decode_op_stats']['memory_bytes_per_token'] * batch_size / 2}. (Example, needs refinement).
B.5. Configuration Parameters for VLLMFramework (within frameworks_to_test[n].config):
* model_profile_id: str (e.g., "Llama2-7B", to look up in model_characteristics_db)
* gpu_id: str (ID of the VirtualComputeDevice to run on)
* block_size: int (KV cache block size in tokens)
* max_num_seqs: int (max concurrent sequences)
* max_num_batched_tokens: int (max tokens processed across all sequences in a scheduling iteration)
* scheduler_iteration_delay_s: float (small delay for the processing loop if no work, e.g., 0.001s)
* bytes_per_token_estimate_for_network: int (for streaming responses)
B.6. Dependencies:
* AbstractLLMFramework
* Core Simulation Engine (for simpy_env)
* VirtualHardwarePlatform
* MetricsCollector
* Request data class
* SequenceState data class (needs to be defined, including request: Request field).
* simpy library (for simpy.AllOf)
Define SequenceState data class (example):
from dataclasses import dataclass, field
from typing import List, Any # Any for Request initially

@dataclass
class SequenceState:
   request_id: str
   request: Any # Should be the Request object
   status: str # e.g., 'WAITING_FOR_PREFILL', 'PREFILLING', 'DECODING', 'COMPLETED'
   prompt_tokens_processed: int = 0
   output_tokens_generated: int = 0
   allocated_kv_blocks: List[Any] = field(default_factory=list) # Store identifiers or count of blocks
   prefill_start_time: float = -1.0 # Simulation time
   # Add other necessary fields