"""vLLM framework simulation implementation."""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import simpy

from .abstract_framework import AbstractLLMFramework
from .models import SequenceState, SessionCacheInfo, GlobalPrefixCacheInfo
from .prefix_utils import hash_token_sequence, find_longest_prefix_match, should_cache_prefix
from ..utils.performance_abstractions import CollectiveCommunicationModels

logger = logging.getLogger(__name__)


class VLLMFramework(AbstractLLMFramework):
    """Simulates the vLLM serving framework with PagedAttention and continuous batching."""
    
    def __init__(
        self,
        framework_id: str,
        simpy_env: simpy.Environment,
        framework_specific_config: Dict[str, Any],
        virtual_hardware: Any,
        metrics_collector: Any,
        model_profile: Dict[str, Any],
    ):
        """Initialize vLLM framework simulation.
        
        Additional config parameters:
            - gpu_id: ID of the virtual GPU to use
            - block_size: KV cache block size in tokens
            - max_num_seqs: Maximum concurrent sequences
            - max_num_batched_tokens: Max tokens per scheduling iteration
            - scheduler_iteration_delay_s: Delay between scheduler iterations
        """
        super().__init__(
            framework_id,
            simpy_env,
            framework_specific_config,
            virtual_hardware,
            metrics_collector,
            model_profile,
        )
        
        # vLLM specific configuration
        self.gpu_id = self.config.get("gpu_id", "gpu0")
        self.block_size = self.config.get("block_size", 16)
        self.max_running_sequences = self.config.get("max_num_seqs", 256)
        
        # Validate model fits in memory (for non-parallel configs)
        if self.parallelism_strategy == 'None':
            self._validate_memory_requirements()
        
        # Dynamic batch size calculation
        if "max_num_batched_tokens" in self.config:
            self.max_batched_tokens_per_iteration = self.config["max_num_batched_tokens"]
        else:
            self.max_batched_tokens_per_iteration = self._calculate_dynamic_batch_size()
        
        # Chunked prefill configuration
        self.enable_chunked_prefill = self.config.get("enable_chunked_prefill", True)
        self.prefill_chunk_size = self.config.get("prefill_chunk_size", 
                                                  min(4096, self.max_batched_tokens_per_iteration))
        
        # For TP, use primary GPU for scheduling but distribute computation
        if self.parallelism_strategy in ['TP', 'TP_PP']:
            self.primary_gpu_id = self.gpu_ids[0]
        else:
            self.primary_gpu_id = self.gpu_id
        
        # PagedAttention KV cache management
        self._init_kv_cache()
        
        # Continuous batching scheduler state
        self.waiting_sequences: List[Any] = []  # Requests waiting for prefill
        self.running_sequences: Dict[str, SequenceState] = {}  # Active sequences
        self.swapped_sequences: List[Any] = []  # For future preemption support
        
        # Prefix caching state
        self.active_sessions_kv_state: Dict[str, SessionCacheInfo] = {}  # Track KV cache per session
        self.enable_prefix_caching = self.config.get("enable_prefix_caching", True)
        
        # Global prefix cache for cross-request caching
        self.global_prefix_store: Dict[str, GlobalPrefixCacheInfo] = {}
        self.enable_cross_request_caching = self.config.get("enable_cross_request_caching", True)
        self.min_prefix_length = self.config.get("min_prefix_cache_length", 50)
        self.max_prefix_cache_size = self.config.get("max_prefix_cache_size", 100)
        self.global_prefix_eviction_policy = self.config.get("prefix_eviction_policy", "lru")
        
        logger.info(
            f"vLLM {framework_id} initialized: "
            f"GPUs={self.gpu_ids if self.parallelism_strategy != 'None' else [self.gpu_id]}, "
            f"parallelism={self.parallelism_strategy}, "
            f"block_size={self.block_size}, "
            f"max_seqs={self.max_running_sequences}, "
            f"max_batched_tokens={self.max_batched_tokens_per_iteration}"
        )
    
    def _validate_memory_requirements(self) -> None:
        """Validate that the model fits in GPU memory."""
        # Get model size
        param_bytes = self.model_profile.get('parameter_bytes_fp16', 0)
        if param_bytes == 0:
            # Estimate from parameter count if not specified
            param_count = self.model_profile.get('parameters', 0)
            param_bytes = param_count * 2  # 2 bytes per parameter for FP16
        
        # Get GPU memory
        device_info = self.virtual_hardware.get_device_info(self.gpu_id)
        if not device_info:
            raise ValueError(f"GPU {self.gpu_id} not found in virtual hardware")
        
        gpu_memory_bytes = device_info.memory_capacity_bytes
        available_memory = gpu_memory_bytes * 0.9  # 90% usable
        
        # Check if model fits
        if param_bytes > available_memory:
            param_gb = param_bytes / 1e9
            available_gb = available_memory / 1e9
            gpu_gb = gpu_memory_bytes / 1e9
            model_id = self.config.get('model_profile_id', 'unknown')
            raise ValueError(
                f"Model {model_id} "
                f"({param_gb:.1f} GB) exceeds available GPU memory "
                f"({available_gb:.1f} GB available out of {gpu_gb:.1f} GB total)"
            )
    
    def _calculate_dynamic_batch_size(self) -> int:
        """Calculate optimal max_num_batched_tokens based on hardware capabilities."""
        # Get hardware specs
        if self.parallelism_strategy in ['TP', 'TP_PP']:
            # Use first GPU as representative
            device_info = self.virtual_hardware.get_device_info(self.gpu_ids[0])
        else:
            device_info = self.virtual_hardware.get_device_info(self.gpu_id)
            
        if not device_info:
            logger.warning("Could not get device info, using default batch size")
            return 4096
            
        # Extract hardware capabilities
        memory_gb = device_info.memory_capacity_bytes / 1e9
        memory_bandwidth_gbs = device_info.memory_gbps
        fp16_tflops = device_info.peak_tflops.get('fp16', 100)
        
        # Model characteristics
        hidden_dim = self.model_profile.get('hidden_size', 4096)
        num_layers = self.model_profile.get('num_layers', 32)
        model_size_gb = self.model_profile.get('parameters', 7e9) * 2 / 1e9  # FP16
        
        # Adjust for parallelism
        if self.parallelism_strategy == 'TP':
            model_size_gb = model_size_gb / self.tp_degree
            # Account for communication overhead
            comm_scaling = min(math.sqrt(self.tp_degree), 4)
        else:
            comm_scaling = 1
            
        # Target iteration time (ms)
        target_iteration_ms = 50
        
        # Calculate limits
        
        # 1. Memory limit for attention matrices and activations
        # Reserve memory for model weights and KV cache
        available_memory_gb = max(0.1, (memory_gb - model_size_gb) * 0.1)  # 10% for activations, min 0.1GB
        # Attention memory scales quadratically for prefill
        # Approximate as sqrt to get token count
        memory_limited = int(math.sqrt(max(1e8, available_memory_gb * 1e9) / (hidden_dim * 4)))
        
        # 2. Bandwidth limit
        # Each token needs to read/write model weights and activations
        bytes_per_token = hidden_dim * num_layers * 4  # Rough estimate
        bandwidth_limited = int(
            (memory_bandwidth_gbs * 1e9 * target_iteration_ms / 1000) / bytes_per_token
        )
        
        # 3. Compute limit
        flops_per_token = self.model_profile.get('prefill_op_stats', {}).get(
            'flops_per_token', 2 * self.model_profile.get('parameters', 7e9)
        )
        if self.parallelism_strategy == 'TP':
            # With TP, each GPU does less work per token
            flops_per_token = flops_per_token / self.tp_degree
        compute_limited = int(
            (fp16_tflops * 1e12 * target_iteration_ms / 1000) / flops_per_token
        )
        
        # Take minimum and apply scaling
        base_limit = min(memory_limited, bandwidth_limited, compute_limited)
        dynamic_limit = int(base_limit * comm_scaling)
        
        # Apply reasonable bounds
        dynamic_limit = max(2048, min(dynamic_limit, 32768))
        
        logger.info(
            f"Calculated dynamic batch size: {dynamic_limit} "
            f"(memory_limited={memory_limited}, bandwidth_limited={bandwidth_limited}, "
            f"compute_limited={compute_limited}, comm_scaling={comm_scaling})"
        )
        
        return dynamic_limit
    
    def _init_kv_cache(self) -> None:
        """Initialize PagedAttention KV cache structures."""
        # Calculate total KV cache blocks available
        # For TP, KV cache is distributed across GPUs
        if self.parallelism_strategy in ['TP', 'TP_PP']:
            # Each GPU in TP group holds 1/tp_degree of the KV cache
            total_memory = 0
            for gpu_id in self.tp_gpu_groups[0]:  # First TP group for now
                device_info = self.virtual_hardware.get_device_info(gpu_id)
                if not device_info:
                    raise ValueError(f"GPU {gpu_id} not found in virtual hardware")
                total_memory += device_info.memory_capacity_bytes
            gpu_memory_for_kv = total_memory * 0.9
        else:
            device_info = self.virtual_hardware.get_device_info(self.gpu_id)
            if not device_info:
                raise ValueError(f"GPU {self.gpu_id} not found in virtual hardware")
            gpu_memory_for_kv = device_info.memory_capacity_bytes * 0.9
        
        bytes_per_block = self._bytes_per_kv_block()
        self.num_gpu_blocks = int(gpu_memory_for_kv / bytes_per_block)
        
        # SimPy container to track free blocks
        self.gpu_blocks_container = simpy.Container(
            self.simpy_env, capacity=self.num_gpu_blocks, init=self.num_gpu_blocks
        )
        
        logger.info(
            f"Initialized KV cache: {self.num_gpu_blocks} blocks, "
            f"{bytes_per_block / 1e6:.1f} MB per block"
        )
    
    def _bytes_per_kv_block(self) -> int:
        """Calculate bytes per KV cache block."""
        return (
            self.block_size *
            self.model_profile["kv_cache_bytes_per_token_per_layer"] *
            self.model_profile["num_layers"]
        )
    
    def _calculate_prompt_kv_blocks(self, num_prompt_tokens: int) -> int:
        """Calculate KV blocks needed for a prompt."""
        return (num_prompt_tokens + self.block_size - 1) // self.block_size
    
    def handle_incoming_request(self, request: Any) -> simpy.Process:
        """Handle incoming request by adding to arrival queue."""
        def _handle_request():
            logger.debug(f"vLLM {self.framework_id} received request {request.request_id}")
            yield self.request_arrival_queue.put(request)
        
        return self.simpy_env.process(_handle_request())
    
    def processing_loop(self) -> simpy.events.Event:
        """Main vLLM continuous batching scheduler loop."""
        logger.info(f"Starting vLLM processing loop for {self.framework_id}")
        
        while True:
            # Part 1: Try to admit new sequences from arrival queue
            newly_admitted = yield from self._admit_new_sequences()
            
            # Part 2: Schedule iteration (Prefill + Decode)
            prefill_batch, decode_batch = self._schedule_iteration()
            
            # Part 3: Execute Prefill Batch
            if prefill_batch:
                yield from self._execute_prefill_batch(prefill_batch)
            
            # Part 4: Execute Decode Batch
            if decode_batch:
                yield from self._execute_decode_batch(decode_batch)
            
            # Part 5: Yield if no work was done
            if not prefill_batch and not decode_batch and not newly_admitted:
                yield self.simpy_env.timeout(
                    self.config.get("scheduler_iteration_delay_s", 0.001)
                )
    
    def _admit_new_sequences(self) -> List[Any]:
        """Try to admit new sequences from the arrival queue."""
        newly_admitted = []
        
        # Check if we have capacity for new sequences
        if len(self.running_sequences) >= self.max_running_sequences:
            return newly_admitted
        
        # Try to admit requests from the queue
        while (
            len(self.request_arrival_queue.items) > 0 and
            len(self.running_sequences) < self.max_running_sequences
        ):
            # Peek at the first request
            request = self.request_arrival_queue.items[0]
            
            # Check for prefix caching opportunities
            cached_prefix_length, num_tokens_to_prefill = self._check_prefix_cache(request)
            
            # Calculate blocks needed only for new tokens
            required_blocks = self._calculate_prompt_kv_blocks(num_tokens_to_prefill)
            
            if self.gpu_blocks_container.level >= required_blocks:
                # Admit the request
                yield self.request_arrival_queue.get()
                
                # Allocate KV blocks only for new tokens
                if required_blocks > 0:
                    yield self.virtual_hardware.allocate_memory(
                        self.gpu_id, required_blocks * self._bytes_per_kv_block()
                    )
                    yield self.gpu_blocks_container.get(required_blocks)
                
                # Create sequence state with prefix caching info
                seq_state = SequenceState(
                    request_id=request.request_id,
                    request=request,
                    status="WAITING_FOR_PREFILL",
                    allocated_kv_blocks=list(range(int(required_blocks))),  # Simplified block tracking
                    cached_prefix_length_used=cached_prefix_length,
                    num_tokens_requiring_prefill=num_tokens_to_prefill,
                    prompt_tokens_fully_processed=0,  # Will be updated after prefill
                    current_prefill_position=cached_prefix_length,  # Start after cached prefix
                )
                
                # Calculate total chunks needed for chunked prefill
                if self.enable_chunked_prefill and num_tokens_to_prefill > self.prefill_chunk_size:
                    seq_state.total_prefill_chunks = math.ceil(num_tokens_to_prefill / self.prefill_chunk_size)
                else:
                    seq_state.total_prefill_chunks = 1
                
                self.running_sequences[request.request_id] = seq_state
                self.waiting_sequences.append(request)
                newly_admitted.append(request)
                
                logger.debug(
                    f"Admitted request {request.request_id}, allocated {required_blocks} KV blocks"
                )
            else:
                # Not enough KV blocks available
                break
        
        return newly_admitted
    
    def _schedule_iteration(self) -> Tuple[List[SequenceState], List[SequenceState]]:
        """Schedule sequences for the current iteration."""
        prefill_batch = []
        decode_batch = []
        current_batched_tokens = 0
        
        # First, add decoding sequences
        for req_id, seq_state in list(self.running_sequences.items()):
            if (
                seq_state.status == "DECODING" and
                current_batched_tokens < self.max_batched_tokens_per_iteration
            ):
                decode_batch.append(seq_state)
                current_batched_tokens += 1
        
        # Then, add prefilling sequences
        for request in list(self.waiting_sequences):
            if request.request_id in self.running_sequences:
                seq_state = self.running_sequences[request.request_id]
                if seq_state.status == "WAITING_FOR_PREFILL" or seq_state.status == "PREFILLING":
                    # Calculate tokens for this iteration (considering chunking)
                    remaining_tokens = seq_state.num_tokens_requiring_prefill - (
                        seq_state.prefill_chunks_completed * self.prefill_chunk_size
                    )
                    
                    if remaining_tokens <= 0:
                        # Prefill already complete, move to decoding
                        seq_state.status = "DECODING"
                        self.waiting_sequences.remove(request)
                        continue
                    
                    # Determine chunk size for this iteration
                    if self.enable_chunked_prefill:
                        tokens_this_chunk = min(
                            remaining_tokens,
                            self.prefill_chunk_size,
                            self.max_batched_tokens_per_iteration - current_batched_tokens
                        )
                    else:
                        tokens_this_chunk = remaining_tokens
                    
                    # Check if we can fit this chunk
                    if tokens_this_chunk > 0 and (current_batched_tokens + tokens_this_chunk) <= self.max_batched_tokens_per_iteration:
                        prefill_batch.append(seq_state)
                        current_batched_tokens += tokens_this_chunk
                        seq_state.status = "PREFILLING"
                        # Don't remove from waiting_sequences if more chunks needed
                        if seq_state.prefill_chunks_completed + 1 >= seq_state.total_prefill_chunks:
                            self.waiting_sequences.remove(request)
                    else:
                        # Can't fit even a small chunk, try next request
                        if not self.enable_chunked_prefill:
                            break  # If no chunking, stop here
                        # With chunking, continue to check other requests
        
        return prefill_batch, decode_batch
    
    def _execute_prefill_batch(self, prefill_batch: List[SequenceState]) -> None:
        """Execute prefill for a batch of sequences."""
        prefill_processes = []
        
        for seq_state in prefill_batch:
            # Calculate tokens for this chunk
            remaining_tokens = seq_state.num_tokens_requiring_prefill - (
                seq_state.prefill_chunks_completed * self.prefill_chunk_size
            )
            
            if self.enable_chunked_prefill:
                tokens_this_chunk = min(remaining_tokens, self.prefill_chunk_size)
            else:
                tokens_this_chunk = remaining_tokens
                
            if tokens_this_chunk > 0:
                # Log prefill start only for first chunk
                if seq_state.prefill_chunks_completed == 0:
                    self.metrics_collector.log_prefill_start(
                        seq_state.request.request_id, 
                        self.simpy_env.now,
                        seq_state.num_tokens_requiring_prefill
                    )
                    seq_state.prefill_start_time = self.simpy_env.now
                
                if self.parallelism_strategy in ['TP', 'TP_PP']:
                    # Execute prefill with tensor parallelism
                    prefill_proc = self._execute_tp_prefill(
                        tokens_this_chunk,
                        seq_state.request.request_id
                    )
                else:
                    # Regular single-GPU prefill
                    task_desc = self._estimate_prefill_ops(
                        tokens_this_chunk,
                        batch_size=1
                    )
                    task_desc["task_id"] = f"{seq_state.request.request_id}_prefill_chunk_{seq_state.prefill_chunks_completed}"
                    prefill_proc = self.virtual_hardware.submit_computation_task(self.gpu_id, task_desc)
                
                prefill_processes.append(prefill_proc)
            else:
                # Full cache hit - no prefill needed
                self.metrics_collector.log_prefix_cache_event(
                    seq_state.request.request_id,
                    self.simpy_env.now,
                    "FULL_HIT_NO_PREFILL_NEEDED",
                    seq_state.cached_prefix_length_used,
                    0
                )
                seq_state.prefill_end_time_sim = self.simpy_env.now
                # Simulate minimal processing time for cache lookup
                minimal_proc = self.simpy_env.timeout(0.0001)
                prefill_processes.append(minimal_proc)
        
        # Wait for all prefills to complete
        if prefill_processes:
            yield simpy.AllOf(self.simpy_env, prefill_processes)
        
        # Update sequence states
        for seq_state in prefill_batch:
            if seq_state.request.request_id in self.running_sequences:
                # Update chunk progress
                seq_state.prefill_chunks_completed += 1
                
                # Update prefill position
                if self.enable_chunked_prefill:
                    tokens_processed_this_iter = min(
                        self.prefill_chunk_size,
                        seq_state.num_tokens_requiring_prefill - 
                        (seq_state.prefill_chunks_completed - 1) * self.prefill_chunk_size
                    )
                    seq_state.current_prefill_position += tokens_processed_this_iter
                else:
                    seq_state.current_prefill_position = (
                        seq_state.cached_prefix_length_used + seq_state.num_tokens_requiring_prefill
                    )
                
                # Check if prefill is complete
                if seq_state.prefill_chunks_completed >= seq_state.total_prefill_chunks:
                    seq_state.status = "DECODING"
                    # Update tokens processed to include both cached and newly prefilled
                    seq_state.prompt_tokens_processed = seq_state.request.prompt_num_tokens
                    seq_state.prompt_tokens_fully_processed = (
                        seq_state.cached_prefix_length_used + seq_state.num_tokens_requiring_prefill
                    )
                    seq_state.prefill_end_time_sim = self.simpy_env.now
                else:
                    # More chunks needed, keep in PREFILLING state
                    seq_state.status = "WAITING_FOR_PREFILL"
                    # Re-add to waiting queue if not already there
                    if seq_state.request not in self.waiting_sequences:
                        self.waiting_sequences.append(seq_state.request)
    
    def _execute_decode_batch(self, decode_batch: List[SequenceState]) -> None:
        """Execute one decode step for a batch of sequences."""
        if not decode_batch:
            return
        
        if self.parallelism_strategy in ['TP', 'TP_PP']:
            # Execute decode with tensor parallelism
            yield from self._execute_tp_decode(len(decode_batch))
        else:
            # Regular single-GPU decode
            task_desc = self._estimate_decode_op(batch_size=len(decode_batch))
            task_desc["task_id"] = f"decode_batch_{self.simpy_env.now}"
            yield self.virtual_hardware.submit_computation_task(self.gpu_id, task_desc)
        
        current_time = self.simpy_env.now
        completed_sequences = []
        
        # Process each sequence in the batch
        for seq_state in decode_batch:
            if seq_state.request.request_id not in self.running_sequences:
                continue
            
            # Log first token if this is the first decode
            if seq_state.output_tokens_generated == 0:
                self.metrics_collector.log_first_token_generated(
                    seq_state.request.request_id,
                    current_time,
                    seq_state.prefill_start_time
                )
                seq_state.first_token_time = current_time
            
            # Generate one token
            seq_state.output_tokens_generated += 1
            self.metrics_collector.log_token_decoded(
                seq_state.request.request_id,
                current_time,
                seq_state.output_tokens_generated
            )
            
            # Simulate streaming response
            if seq_state.request.streaming_response:
                token_data_size = self.config.get("bytes_per_token_estimate_for_network", 2)
                # Fire and forget network transfer for streaming
                self.virtual_hardware.submit_network_transfer_task(
                    self.gpu_id,
                    "client_node_0",
                    token_data_size
                )
            
            # Check completion conditions
            if (
                seq_state.output_tokens_generated >= seq_state.request.max_output_tokens or
                self._check_eos_condition(seq_state)
            ):
                self.metrics_collector.log_request_completed(
                    seq_state.request.request_id,
                    current_time,
                    seq_state.output_tokens_generated,
                    "SUCCESS"
                )
                completed_sequences.append(seq_state.request.request_id)
            else:
                # Check if we need a new KV block
                total_tokens = seq_state.prompt_tokens_processed + seq_state.output_tokens_generated
                if total_tokens % self.block_size == 0:
                    # Try to allocate a new block
                    can_allocate = self.gpu_blocks_container.get(1)
                    block_allocated = yield can_allocate | self.simpy_env.timeout(0)
                    
                    if can_allocate not in block_allocated:
                        # Out of KV blocks
                        self.metrics_collector.log_request_completed(
                            seq_state.request.request_id,
                            current_time,
                            seq_state.output_tokens_generated,
                            "OOM_KV_BLOCK"
                        )
                        completed_sequences.append(seq_state.request.request_id)
                        yield from self._release_sequence_resources(seq_state)
        
        # Clean up completed sequences
        for req_id in completed_sequences:
            if req_id in self.running_sequences:
                seq_state = self.running_sequences.pop(req_id)
                # Update session cache for conversational requests
                if seq_state.request.session_id and self.enable_prefix_caching:
                    self._update_session_cache(seq_state)
                # Consider adding to global cache
                if self.enable_cross_request_caching:
                    self._maybe_add_to_global_cache(seq_state)
                yield from self._release_sequence_resources(seq_state)
    
    def _release_sequence_resources(self, seq_state: SequenceState) -> simpy.events.Event:
        """Release resources held by a sequence."""
        num_blocks = len(seq_state.allocated_kv_blocks)
        if num_blocks > 0:
            yield self.gpu_blocks_container.put(num_blocks)
            yield self.virtual_hardware.free_memory(
                self.gpu_id,
                num_blocks * self._bytes_per_kv_block()
            )
            logger.debug(f"Released {num_blocks} KV blocks from {seq_state.request.request_id}")
    
    def _check_eos_condition(self, seq_state: SequenceState) -> bool:
        """Check if end-of-sequence condition is met."""
        # Placeholder - in real vLLM this would check for EOS token
        return False
    
    def _check_prefix_cache(self, request: Any) -> Tuple[int, int]:
        """Check if request can use cached prefix and return (cached_length, tokens_to_prefill)."""
        if not self.enable_prefix_caching:
            return 0, request.prompt_num_tokens
        
        cached_prefix_length = 0
        num_tokens_to_prefill = request.prompt_num_tokens
        cache_hit_type = None
        
        # Priority 1: Check for conversational prefix cache hit
        if request.is_conversational_turn and request.session_id in self.active_sessions_kv_state:
            session_cache_info = self.active_sessions_kv_state[request.session_id]
            
            # Assume the prompt includes the full history (cached content + new input)
            if request.prompt_num_tokens > session_cache_info.total_tokens_in_cache:
                # We can reuse the cached prefix
                cached_prefix_length = session_cache_info.total_tokens_in_cache
                num_tokens_to_prefill = request.prompt_num_tokens - cached_prefix_length
                cache_hit_type = "CONVERSATIONAL_HIT"
                
                logger.debug(
                    f"Conversational cache hit for {request.request_id}: "
                    f"reusing {cached_prefix_length} tokens, prefilling {num_tokens_to_prefill} new tokens"
                )
            else:
                # Unexpected: conversational turn with shorter prompt
                cache_hit_type = "CONVERSATIONAL_MISS_UNEXPECTED_PROMPT"
                logger.warning(
                    f"Conversational turn {request.request_id} has shorter prompt than cached history"
                )
        
        # Priority 2: Check for cross-request prefix cache hit (if no conversational hit)
        if (cache_hit_type is None and self.enable_cross_request_caching and 
            hasattr(request, 'prompt_tokens') and request.prompt_tokens):
            
            # Find longest matching prefix in global cache
            matching_hash, prefix_length = find_longest_prefix_match(
                request.prompt_tokens,
                self.global_prefix_store,
                self.min_prefix_length
            )
            
            if matching_hash:
                # Found a matching prefix
                cached_prefix_length = prefix_length
                num_tokens_to_prefill = request.prompt_num_tokens - cached_prefix_length
                cache_hit_type = "CROSS_REQUEST_HIT"
                
                # Update access statistics
                prefix_info = self.global_prefix_store[matching_hash]
                prefix_info.reference_count += 1
                prefix_info.last_access_time = self.simpy_env.now
                prefix_info.access_count += 1
                
                logger.debug(
                    f"Cross-request cache hit for {request.request_id}: "
                    f"reusing {cached_prefix_length} tokens from hash {matching_hash[:8]}..."
                )
        
        # No cache hit
        if cache_hit_type is None:
            cache_hit_type = "MISS_FULL"
        
        # Log the cache event
        self.metrics_collector.log_prefix_cache_event(
            request.request_id,
            self.simpy_env.now,
            cache_hit_type,
            cached_prefix_length,
            num_tokens_to_prefill
        )
        
        return cached_prefix_length, num_tokens_to_prefill
    
    def _update_session_cache(self, completed_seq_state: SequenceState) -> None:
        """Update session cache info after a sequence completes."""
        session_id = completed_seq_state.request.session_id
        if not session_id:
            return
        
        # Create new session cache info
        new_cache_info = SessionCacheInfo(
            session_id=session_id,
            total_tokens_in_cache=(
                completed_seq_state.prompt_tokens_fully_processed + 
                completed_seq_state.output_tokens_generated
            ),
            prompt_part_length=completed_seq_state.prompt_tokens_fully_processed,
            response_part_length=completed_seq_state.output_tokens_generated,
            associated_sequence_id=completed_seq_state.request.request_id,
            kv_block_ids=completed_seq_state.allocated_kv_blocks.copy(),
            last_update_time=self.simpy_env.now
        )
        
        self.active_sessions_kv_state[session_id] = new_cache_info
        
        logger.debug(
            f"Updated session cache for {session_id}: "
            f"{new_cache_info.total_tokens_in_cache} tokens cached"
        )
    
    def _maybe_add_to_global_cache(self, seq_state: SequenceState) -> None:
        """Consider adding a prefix to the global cache after processing."""
        if not self.enable_cross_request_caching:
            return
        
        request = seq_state.request
        if not hasattr(request, 'prompt_tokens') or not request.prompt_tokens:
            return
        
        # Only cache if this was a cache miss and the prefix is long enough
        if seq_state.cached_prefix_length_used > 0:
            return  # This request already used a cached prefix
        
        prompt_length = len(request.prompt_tokens)
        if not should_cache_prefix(
            prompt_length, 
            self.min_prefix_length,
            self.max_prefix_cache_size,
            len(self.global_prefix_store)
        ):
            return
        
        # Check for multiple potential prefix lengths (e.g., 50, 100, 150 tokens)
        prefix_lengths_to_check = [
            length for length in [50, 100, 200, 300, 500]
            if length <= prompt_length
        ]
        
        for prefix_length in prefix_lengths_to_check:
            prefix_hash = hash_token_sequence(request.prompt_tokens, prefix_length)
            
            # Skip if already cached
            if prefix_hash in self.global_prefix_store:
                continue
            
            # Add to global cache
            self._add_prefix_to_global_cache(
                prefix_hash,
                prefix_length,
                request.prompt_tokens[:prefix_length],
                seq_state
            )
            break  # Only cache one prefix length per request
    
    def _add_prefix_to_global_cache(
        self, 
        prefix_hash: str, 
        prefix_length: int,
        prefix_tokens: List[int],
        seq_state: SequenceState
    ) -> None:
        """Add a new prefix to the global cache."""
        # Evict if at capacity
        if len(self.global_prefix_store) >= self.max_prefix_cache_size:
            self._evict_from_global_cache()
        
        # Create cache entry
        cache_info = GlobalPrefixCacheInfo(
            prefix_hash=prefix_hash,
            prefix_length=prefix_length,
            kv_block_ids=[],  # Simplified: not tracking actual blocks
            reference_count=1,
            last_access_time=self.simpy_env.now,
            creation_time=self.simpy_env.now,
            access_count=0,
            prefix_tokens=prefix_tokens if self.config.get("store_prefix_tokens", False) else None
        )
        
        self.global_prefix_store[prefix_hash] = cache_info
        
        logger.debug(
            f"Added prefix to global cache: hash={prefix_hash[:8]}..., "
            f"length={prefix_length}, cache_size={len(self.global_prefix_store)}"
        )
    
    def _evict_from_global_cache(self) -> None:
        """Evict entries from global cache based on policy (LRU by default)."""
        if not self.global_prefix_store:
            return
        
        if self.global_prefix_eviction_policy == "lru":
            # Find least recently used entry with reference count 0
            eviction_candidate = None
            oldest_access_time = float('inf')
            
            for prefix_hash, cache_info in self.global_prefix_store.items():
                if cache_info.reference_count == 0 and cache_info.last_access_time < oldest_access_time:
                    eviction_candidate = prefix_hash
                    oldest_access_time = cache_info.last_access_time
            
            if eviction_candidate:
                evicted = self.global_prefix_store.pop(eviction_candidate)
                logger.debug(
                    f"Evicted prefix from global cache: hash={eviction_candidate[:8]}..., "
                    f"length={evicted.prefix_length}, access_count={evicted.access_count}"
                )
            else:
                logger.warning("Global prefix cache full but no entries eligible for eviction")
    
    def _decrement_global_prefix_ref_count(self, seq_state: SequenceState) -> None:
        """Decrement reference count when a sequence using a global prefix completes."""
        # In the simplified model, we track which prefix was used via the sequence state
        # In a real implementation, this would be more sophisticated
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get current framework status."""
        return {
            "framework_id": self.framework_id,
            "lod": self.lod,
            "queue_length": len(self.request_arrival_queue.items),
            "waiting_sequences": len(self.waiting_sequences),
            "running_sequences": len(self.running_sequences),
            "kv_blocks_used": self.num_gpu_blocks - self.gpu_blocks_container.level,
            "kv_blocks_total": self.num_gpu_blocks,
        }
    
    def _execute_tp_prefill(self, num_tokens: int, request_id: str) -> simpy.Process:
        """Execute prefill with tensor parallelism."""
        def _tp_prefill():
            # Get the current TP GPU group (for now, just use the first one)
            tp_gpu_group = self.tp_gpu_groups[0]
            
            if self.lod == "medium":
                # Use macro operations for entire prefill
                # Get aggregated operation stats
                ops = self._estimate_prefill_ops(num_tokens, batch_size=1)
                
                # Distribute across TP GPUs
                tp_degree = len(tp_gpu_group)
                sharded_ops = {
                    'flops_required_fp16': ops['flops_required_fp16'] / tp_degree,
                    'memory_read_bytes': ops['memory_read_bytes'] / tp_degree,
                    'memory_write_bytes': ops['memory_write_bytes'] / tp_degree,
                    'is_memory_bound_hint': ops.get('is_memory_bound_hint', False)
                }
                
                # Submit parallel compute tasks
                compute_events = []
                for i, gpu_id in enumerate(tp_gpu_group):
                    task_desc = sharded_ops.copy()
                    task_desc['task_id'] = f"{request_id}_tp{i}_macro"
                    compute_event = self.virtual_hardware.submit_computation_task(gpu_id, task_desc)
                    compute_events.append(compute_event)
                
                # Wait for all shards
                yield simpy.AllOf(self.simpy_env, compute_events)
                
                # Single collective for entire prefill
                if tp_degree > 1:
                    yield from self._simulate_tp_collective(
                        'AllReduce',
                        tp_gpu_group,
                        batch_size=1,
                        sequence_length=num_tokens,
                        op_type='prefill_macro'
                    )
            else:
                # High LoD: process each layer
                num_layers = self.model_profile.get('num_layers', 32)
                for layer_idx in range(num_layers):
                    # Alternate between attention and MLP layers
                    if layer_idx % 2 == 0:
                        layer_type = 'attention'
                    else:
                        layer_type = 'mlp'
                    
                    layer_stats = self._get_layer_stats(layer_type)
                    if not layer_stats:
                        # Fallback to regular prefill stats
                        layer_stats = {
                            'flops_per_token_prefill': self.model_profile['prefill_op_stats']['flops_per_token'] / num_layers,
                            'memory_bytes_per_token_prefill': self.model_profile['prefill_op_stats']['memory_bytes_per_token'] / num_layers,
                            'tp_collective_type': 'AllReduce'
                        }
                    
                    # Dispatch TP sharded operation
                    yield from self._dispatch_tp_shardable_operation(
                        layer_stats,
                        batch_size=1,
                        sequence_length=num_tokens,
                        op_type=f'{layer_type}_prefill',
                        tp_collective_type=layer_stats.get('tp_collective_type', 'AllReduce'),
                        tp_gpu_group=tp_gpu_group,
                        request_id=f"{request_id}_layer_{layer_idx}"
                    )
        
        return self.simpy_env.process(_tp_prefill())
    
    def _execute_tp_decode(self, batch_size: int) -> simpy.events.Event:
        """Execute decode with tensor parallelism."""
        # Get the current TP GPU group
        tp_gpu_group = self.tp_gpu_groups[0]
        
        if self.lod == "medium":
            # Use macro operations for entire decode step
            # Get aggregated operation stats
            ops = self._estimate_decode_op(batch_size)
            
            # Distribute across TP GPUs
            tp_degree = len(tp_gpu_group)
            sharded_ops = {
                'flops_required_fp16': ops['flops_required_fp16'] / tp_degree,
                'memory_read_bytes': ops['memory_read_bytes'] / tp_degree,
                'memory_write_bytes': ops['memory_write_bytes'] / tp_degree,
                'is_memory_bound_hint': ops.get('is_memory_bound_hint', True)
            }
            
            # Submit parallel compute tasks
            compute_events = []
            for i, gpu_id in enumerate(tp_gpu_group):
                task_desc = sharded_ops.copy()
                task_desc['task_id'] = f"decode_batch_{self.simpy_env.now}_tp{i}_macro"
                compute_event = self.virtual_hardware.submit_computation_task(gpu_id, task_desc)
                compute_events.append(compute_event)
            
            # Wait for all shards
            yield simpy.AllOf(self.simpy_env, compute_events)
            
            # Single collective for entire decode
            if tp_degree > 1:
                yield from self._simulate_tp_collective(
                    'AllReduce',
                    tp_gpu_group,
                    batch_size=batch_size,
                    sequence_length=1,
                    op_type='decode_macro'
                )
        else:
            # High LoD: process each layer
            num_layers = self.model_profile.get('num_layers', 32)
            for layer_idx in range(num_layers):
                # Alternate between attention and MLP layers
                if layer_idx % 2 == 0:
                    layer_type = 'attention'
                else:
                    layer_type = 'mlp'
                
                layer_stats = self._get_layer_stats(layer_type)
                if not layer_stats:
                    # Fallback to regular decode stats
                    layer_stats = {
                        'flops_per_token_decode': self.model_profile['decode_op_stats']['flops_per_token'] / num_layers,
                        'memory_bytes_per_token_decode': self.model_profile['decode_op_stats']['memory_bytes_per_token'] / num_layers,
                        'tp_collective_type': 'AllReduce'
                    }
                
                # Dispatch TP sharded operation
                yield from self._dispatch_tp_shardable_operation(
                    layer_stats,
                    batch_size=batch_size,
                    sequence_length=1,  # Decode processes one token at a time
                    op_type=f'{layer_type}_decode',
                    tp_collective_type=layer_stats.get('tp_collective_type', 'AllReduce'),
                    tp_gpu_group=tp_gpu_group,
                    request_id=f"decode_batch_{self.simpy_env.now}_layer_{layer_idx}"
                )
    
    def _dispatch_tp_shardable_operation(
        self,
        base_op_stats: Dict[str, Any],
        batch_size: int,
        sequence_length: int,
        op_type: str,
        tp_collective_type: str,
        tp_gpu_group: List[str],
        request_id: str
    ) -> simpy.events.Event:
        """Dispatch a tensor-parallel shardable operation."""
        tp_degree = len(tp_gpu_group)
        
        # Calculate sharded compute for each GPU
        if 'prefill' in op_type:
            base_flops = base_op_stats.get('flops_per_token_prefill', 1e9)
            base_memory = base_op_stats.get('memory_bytes_per_token_prefill', 1e6)
        else:
            base_flops = base_op_stats.get('flops_per_token_decode', 1e9)
            base_memory = base_op_stats.get('memory_bytes_per_token_decode', 1e6)
        
        sharded_flops = base_flops / tp_degree
        sharded_memory = base_memory / tp_degree
        
        # Create task description for each GPU
        task_description = {
            'flops_required_fp16': sharded_flops * sequence_length * batch_size,
            'memory_read_bytes': sharded_memory * sequence_length * batch_size,
            'memory_write_bytes': sharded_memory * sequence_length * batch_size / 2,
            'is_memory_bound_hint': 'decode' in op_type,
        }
        
        # Submit parallel compute tasks
        compute_events = []
        for i, gpu_id in enumerate(tp_gpu_group):
            task_desc = task_description.copy()
            task_desc['task_id'] = f"{request_id}_tp{i}"
            compute_event = self.virtual_hardware.submit_computation_task(gpu_id, task_desc)
            compute_events.append(compute_event)
        
        # Wait for all shards to compute
        yield simpy.AllOf(self.simpy_env, compute_events)
        
        # Simulate collective communication
        if tp_degree > 1:
            yield from self._simulate_tp_collective(
                tp_collective_type,
                tp_gpu_group,
                batch_size,
                sequence_length,
                op_type
            )
    
    def _simulate_tp_collective(
        self,
        collective_type: str,
        gpu_group: List[str],
        batch_size: int,
        sequence_length: int,
        op_type: str
    ) -> simpy.events.Event:
        """Simulate tensor parallel collective communication."""
        # Estimate data size for collective
        hidden_size = self.model_profile.get('hidden_size', 4096)
        bytes_per_element = 2  # FP16
        
        # For attention/MLP output, we need to all-reduce activations
        activation_size = batch_size * sequence_length * hidden_size * bytes_per_element
        
        num_gpus = len(gpu_group)
        if num_gpus <= 1:
            return
        
        if self.lod == "medium":
            # Use analytical model for collective communication
            # Get network link parameters (assume uniform links between GPUs)
            # Try to find a link between first two GPUs as representative
            link_bandwidth_bps = 100e9  # Default 100 Gbps
            link_latency_s = 1e-6  # Default 1 microsecond
            
            if len(gpu_group) >= 2:
                # Try to access the network link from virtual hardware
                link_key = (gpu_group[0], gpu_group[1])
                if hasattr(self.virtual_hardware, 'network_links'):
                    link = self.virtual_hardware.network_links.get(link_key)
                    if not link:
                        # Try reverse direction
                        link_key = (gpu_group[1], gpu_group[0])
                        link = self.virtual_hardware.network_links.get(link_key)
                    
                    if link:
                        link_bandwidth_bps = link.bandwidth_bps
                        link_latency_s = link.latency_s
            
            # Calculate collective time using analytical model
            collective_time = CollectiveCommunicationModels.get_collective_time(
                collective_type,
                activation_size,
                num_gpus,
                link_bandwidth_bps,
                link_latency_s
            )
            
            # Simply wait for the calculated time
            yield self.simpy_env.timeout(collective_time)
            
        else:
            # High LoD: detailed simulation
            # For ring allreduce, each GPU sends/receives data in a ring pattern
            # Simplified: just simulate point-to-point transfers in a ring
            collective_events = []
            
            for i in range(num_gpus):
                source_gpu = gpu_group[i]
                dest_gpu = gpu_group[(i + 1) % num_gpus]
                
                # Each step transfers 1/N of the data
                transfer_size = activation_size / num_gpus
                
                transfer_event = self.virtual_hardware.submit_network_transfer_task(
                    source_gpu,
                    dest_gpu,
                    transfer_size
                )
                collective_events.append(transfer_event)
            
            # In a real ring allreduce, these would be pipelined
            # For simplicity, we wait for all transfers
            yield simpy.AllOf(self.simpy_env, collective_events)