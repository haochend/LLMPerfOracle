"""vLLM framework simulation implementation."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import simpy

from .abstract_framework import AbstractLLMFramework
from .models import SequenceState, SessionCacheInfo

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
        self.max_batched_tokens_per_iteration = self.config.get("max_num_batched_tokens", 4096)
        
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
        self.active_sessions_kv_state: Dict[str, 'SessionCacheInfo'] = {}  # Track KV cache per session
        self.enable_prefix_caching = self.config.get("enable_prefix_caching", True)
        
        logger.info(
            f"vLLM {framework_id} initialized: "
            f"GPUs={self.gpu_ids if self.parallelism_strategy != 'None' else [self.gpu_id]}, "
            f"parallelism={self.parallelism_strategy}, "
            f"block_size={self.block_size}, "
            f"max_seqs={self.max_running_sequences}"
        )
    
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
                    allocated_kv_blocks=list(range(required_blocks)),  # Simplified block tracking
                    cached_prefix_length_used=cached_prefix_length,
                    num_tokens_requiring_prefill=num_tokens_to_prefill,
                    prompt_tokens_fully_processed=0,  # Will be updated after prefill
                )
                
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
                if seq_state.status == "WAITING_FOR_PREFILL":
                    # Use actual tokens requiring prefill, not total prompt tokens
                    tokens_for_prefill = seq_state.num_tokens_requiring_prefill
                    if (current_batched_tokens + tokens_for_prefill) <= self.max_batched_tokens_per_iteration:
                        prefill_batch.append(seq_state)
                        current_batched_tokens += tokens_for_prefill
                        self.waiting_sequences.remove(request)
                        seq_state.status = "PREFILLING"
                    else:
                        break
        
        return prefill_batch, decode_batch
    
    def _execute_prefill_batch(self, prefill_batch: List[SequenceState]) -> None:
        """Execute prefill for a batch of sequences."""
        prefill_processes = []
        
        for seq_state in prefill_batch:
            if seq_state.num_tokens_requiring_prefill > 0:
                # Log prefill start with actual tokens to prefill
                self.metrics_collector.log_prefill_start(
                    seq_state.request.request_id, 
                    self.simpy_env.now,
                    seq_state.num_tokens_requiring_prefill
                )
                seq_state.prefill_start_time = self.simpy_env.now
                
                if self.parallelism_strategy in ['TP', 'TP_PP']:
                    # Execute prefill with tensor parallelism
                    prefill_proc = self._execute_tp_prefill(
                        seq_state.num_tokens_requiring_prefill,  # Use actual tokens to prefill
                        seq_state.request.request_id
                    )
                else:
                    # Regular single-GPU prefill
                    task_desc = self._estimate_prefill_ops(
                        seq_state.num_tokens_requiring_prefill,  # Use actual tokens to prefill
                        batch_size=1
                    )
                    task_desc["task_id"] = f"{seq_state.request.request_id}_prefill"
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
                seq_state.status = "DECODING"
                # Update tokens processed to include both cached and newly prefilled
                seq_state.prompt_tokens_processed = seq_state.request.prompt_num_tokens
                seq_state.prompt_tokens_fully_processed = (
                    seq_state.cached_prefix_length_used + seq_state.num_tokens_requiring_prefill
                )
    
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
        
        # Check for conversational prefix cache hit
        if request.is_conversational_turn and request.session_id in self.active_sessions_kv_state:
            session_cache_info = self.active_sessions_kv_state[request.session_id]
            
            # Assume the prompt includes the full history (cached content + new input)
            if request.prompt_num_tokens > session_cache_info.total_tokens_in_cache:
                # We can reuse the cached prefix
                cached_prefix_length = session_cache_info.total_tokens_in_cache
                num_tokens_to_prefill = request.prompt_num_tokens - cached_prefix_length
                
                # Log cache hit
                self.metrics_collector.log_prefix_cache_event(
                    request.request_id,
                    self.simpy_env.now,
                    "CONVERSATIONAL_HIT",
                    cached_prefix_length,
                    num_tokens_to_prefill
                )
                logger.debug(
                    f"Prefix cache hit for {request.request_id}: "
                    f"reusing {cached_prefix_length} tokens, prefilling {num_tokens_to_prefill} new tokens"
                )
            else:
                # Unexpected: conversational turn with shorter prompt
                self.metrics_collector.log_prefix_cache_event(
                    request.request_id,
                    self.simpy_env.now,
                    "CONVERSATIONAL_MISS_UNEXPECTED_PROMPT",
                    0,
                    request.prompt_num_tokens
                )
                logger.warning(
                    f"Conversational turn {request.request_id} has shorter prompt than cached history"
                )
        else:
            # No conversational cache hit
            self.metrics_collector.log_prefix_cache_event(
                request.request_id,
                self.simpy_env.now,
                "MISS_FULL",
                0,
                request.prompt_num_tokens
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
    
    def get_status(self) -> Dict[str, Any]:
        """Get current framework status."""
        return {
            "framework_id": self.framework_id,
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
            
            # Process each layer
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
        
        # Process each layer
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
        
        # Simplified ring collective simulation
        num_gpus = len(gpu_group)
        if num_gpus <= 1:
            return
        
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