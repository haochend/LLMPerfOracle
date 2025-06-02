"""Parallel vLLM framework simulation with support for Pipeline Parallelism."""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import simpy

from .vllm_framework import VLLMFramework
from .models import SequenceState

logger = logging.getLogger(__name__)


@dataclass
class MicrobatchState:
    """State for tracking microbatches in pipeline parallelism."""
    request_id: str
    microbatch_idx: int
    current_stage: int
    data_size_tokens: int
    is_prefill: bool
    arrival_time_at_stage: float = 0.0
    stage_outputs: Dict[int, Any] = field(default_factory=dict)


class ParallelVLLMFramework(VLLMFramework):
    """vLLM framework simulation with Pipeline Parallelism support."""
    
    def __init__(
        self,
        framework_id: str,
        simpy_env: simpy.Environment,
        framework_specific_config: Dict[str, Any],
        virtual_hardware: Any,
        metrics_collector: Any,
        model_profile: Dict[str, Any],
    ):
        """Initialize parallel vLLM framework simulation."""
        super().__init__(
            framework_id,
            simpy_env,
            framework_specific_config,
            virtual_hardware,
            metrics_collector,
            model_profile,
        )
        
        # Pipeline parallelism specific validation
        if self.parallelism_strategy in ['PP', 'TP_PP']:
            # Check if stages exceed layers
            num_layers = self.model_profile.get('num_layers', 0)
            if self.pp_stages > num_layers:
                raise ValueError(f"Cannot have more PP stages ({self.pp_stages}) than model layers ({num_layers})")
            
            # Validate microbatch count
            if self.num_microbatches <= 0:
                raise ValueError(f"num_microbatches_per_request must be positive, got {self.num_microbatches}")
            
            self._init_pipeline_parallelism()
    
    def _init_pipeline_parallelism(self):
        """Initialize pipeline parallelism structures."""
        # Create input queues for each pipeline stage
        self.stage_input_queues = [
            simpy.Store(self.simpy_env) for _ in range(self.pp_stages)
        ]
        
        # Track microbatch states
        self.active_microbatches: Dict[str, List[MicrobatchState]] = {}
        
        # Track sequences pending deletion (completed but still have active microbatches)
        self.sequences_pending_deletion: Dict[str, Any] = {}
        
        # Track prefill microbatch counts to know when prefill is complete
        self.prefill_microbatch_count: Dict[str, int] = {}
        
        # Start pipeline stage worker processes
        for stage_idx in range(self.pp_stages):
            self.simpy_env.process(self._pipeline_stage_worker_process(stage_idx))
        
        logger.info(
            f"Initialized pipeline parallelism with {self.pp_stages} stages"
        )
    
    def _init_kv_cache(self) -> None:
        """Initialize KV cache for pipeline parallelism - only for assigned layers."""
        if self.parallelism_strategy in ['PP', 'TP_PP']:
            # For PP, each stage only needs KV cache for its assigned layers
            # The _bytes_per_kv_block method is already overridden to account for this
            
            # Get total memory available for this stage
            if self.parallelism_strategy == 'TP_PP':
                # For TP+PP, sum memory across TP group for this stage
                total_memory = 0
                # Use first stage's GPU assignment as representative
                for gpu_id in self.pp_stage_to_gpus[0]:
                    device_info = self.virtual_hardware.get_device_info(gpu_id)
                    if not device_info:
                        raise ValueError(f"GPU {gpu_id} not found in virtual hardware")
                    total_memory += device_info.memory_capacity_bytes
                gpu_memory_for_kv = total_memory * 0.9
            else:
                # For PP only, use memory from first GPU of first stage
                gpu_id = self.pp_stage_to_gpus[0][0]
                device_info = self.virtual_hardware.get_device_info(gpu_id)
                if not device_info:
                    raise ValueError(f"GPU {gpu_id} not found in virtual hardware")
                gpu_memory_for_kv = device_info.memory_capacity_bytes * 0.9
            
            # Use the overridden _bytes_per_kv_block which accounts for layers per stage
            bytes_per_block = self._bytes_per_kv_block()
            self.num_gpu_blocks = int(gpu_memory_for_kv / bytes_per_block)
            
            # Create container with adjusted capacity
            self.gpu_blocks_container = simpy.Container(
                self.simpy_env, capacity=self.num_gpu_blocks, init=self.num_gpu_blocks
            )
            
            total_layers = self.model_profile.get('num_layers', 32)
            layers_per_stage = total_layers / self.pp_stages
            
            logger.info(
                f"PP KV cache initialized: {self.num_gpu_blocks} blocks for {layers_per_stage:.1f} layers "
                f"({bytes_per_block / 1024:.1f} KB per block)"
            )
        else:
            # Fall back to parent implementation for non-PP
            super()._init_kv_cache()
    
    def _bytes_per_kv_block(self) -> int:
        """Calculate bytes per KV cache block for PP - only for assigned layers."""
        if self.parallelism_strategy in ['PP', 'TP_PP']:
            # For PP, only count layers assigned to this pipeline stage
            total_layers = self.model_profile.get('num_layers', 32)
            layers_per_stage = total_layers / self.pp_stages
            
            return int(
                self.block_size *
                self.model_profile["kv_cache_bytes_per_token_per_layer"] *
                layers_per_stage
            )
        else:
            # Fall back to parent implementation
            return super()._bytes_per_kv_block()
    
    def processing_loop(self) -> simpy.events.Event:
        """Main processing loop with pipeline parallelism support."""
        if self.parallelism_strategy in ['PP', 'TP_PP']:
            # For PP, the main loop just handles request admission and microbatch creation
            return self._pp_processing_loop()
        else:
            # Fall back to regular vLLM processing
            return super().processing_loop()
    
    def _pp_processing_loop(self) -> simpy.events.Event:
        """Pipeline parallel processing loop."""
        logger.info(f"Starting PP processing loop for {self.framework_id} with LoD: {self.lod}")
        
        while True:
            # Try to admit new sequences from arrival queue
            newly_admitted = yield from self._admit_new_sequences()
            
            if newly_admitted:
                logger.debug(f"PP admitted {len(newly_admitted)} new requests")
            
            # Create microbatches for newly admitted requests
            for request in newly_admitted:
                yield from self._create_and_dispatch_microbatches(request)
            
            # Small delay if no work
            if not newly_admitted:
                yield self.simpy_env.timeout(
                    self.config.get("scheduler_iteration_delay_s", 0.001)
                )
    
    def _create_and_dispatch_microbatches(self, request: Any) -> simpy.events.Event:
        """Create microbatches for a request and dispatch to first stage."""
        # Set prefill start time for the sequence
        seq_state = self.running_sequences.get(request.request_id)
        if seq_state and seq_state.prefill_start_time < 0:
            seq_state.prefill_start_time = self.simpy_env.now
        
        num_microbatches = self.num_microbatches
        tokens_per_microbatch = request.prompt_num_tokens // num_microbatches
        remainder_tokens = request.prompt_num_tokens % num_microbatches
        
        microbatch_states = []
        
        for mb_idx in range(num_microbatches):
            # Distribute remainder tokens among first microbatches
            mb_tokens = tokens_per_microbatch
            if mb_idx < remainder_tokens:
                mb_tokens += 1
            
            mb_state = MicrobatchState(
                request_id=request.request_id,
                microbatch_idx=mb_idx,
                current_stage=0,
                data_size_tokens=mb_tokens,
                is_prefill=True,
                arrival_time_at_stage=self.simpy_env.now
            )
            
            microbatch_states.append(mb_state)
            
            # Dispatch to first stage
            yield self.stage_input_queues[0].put(mb_state)
        
        # Track active microbatch indices for this request
        self.active_microbatches[request.request_id] = set(range(num_microbatches))
        
        # Track that this is a prefill phase
        self.prefill_microbatch_count[request.request_id] = num_microbatches
        
        logger.debug(
            f"Created {num_microbatches} microbatches for request {request.request_id}"
        )
    
    def _pipeline_stage_worker_process(self, stage_idx: int) -> simpy.events.Event:
        """Worker process for a pipeline stage."""
        logger.info(f"Starting pipeline stage worker {stage_idx}")
        
        while True:
            # Get microbatch from input queue
            microbatch = yield self.stage_input_queues[stage_idx].get()
            
            # Set current TP GPU group for this stage
            if self.parallelism_strategy == 'TP_PP':
                self.current_tp_gpu_group = self.pp_stage_to_gpus[stage_idx]
            else:
                self.current_tp_gpu_group = [self.pp_stage_to_gpus[stage_idx][0]]
            
            # Process layers assigned to this stage
            yield from self._process_stage_layers(stage_idx, microbatch)
            
            # Transfer to next stage or complete
            if stage_idx < self.pp_stages - 1:
                # Transfer activations to next stage
                yield from self._transfer_to_next_stage(stage_idx, microbatch)
            else:
                # Last stage - handle completion
                yield from self._complete_microbatch(microbatch)
    
    def _process_stage_layers(
        self, stage_idx: int, microbatch: MicrobatchState
    ) -> simpy.events.Event:
        """Process layers assigned to a pipeline stage."""
        start_layer, end_layer = self.stage_layer_ranges[stage_idx]
        
        logger.debug(
            f"Stage {stage_idx} processing layers {start_layer}-{end_layer} "
            f"for microbatch {microbatch.request_id}_{microbatch.microbatch_idx}"
        )
        
        # Get sequence state - check both running and pending deletion
        seq_state = self.running_sequences.get(microbatch.request_id)
        if not seq_state:
            seq_state = self.sequences_pending_deletion.get(microbatch.request_id)
            if not seq_state:
                # This can happen if microbatches arrive out of order
                # Just skip processing this microbatch
                logger.debug(f"Sequence {microbatch.request_id} already completed, skipping microbatch")
                yield self.simpy_env.timeout(0)
                return
        
        # Check if we should use aggregated operations for medium LoD
        if self.lod == "medium":
            # Use aggregated operations for the entire stage
            logger.debug(f"Using aggregated operations for stage {stage_idx} (LoD: {self.lod})")
            yield from self._process_stage_layers_aggregated(
                stage_idx, microbatch, start_layer, end_layer
            )
        else:
            # High LoD: Process each layer individually
            logger.debug(f"Using detailed operations for stage {stage_idx} (LoD: {self.lod})")
            yield from self._process_stage_layers_detailed(
                stage_idx, microbatch, start_layer, end_layer
            )
    
    def _process_stage_layers_aggregated(
        self, stage_idx: int, microbatch: MicrobatchState, 
        start_layer: int, end_layer: int
    ) -> simpy.events.Event:
        """Process stage layers using aggregated operations (medium LoD)."""
        from ..utils.performance_abstractions import MacroOperations
        
        # Calculate aggregated operations for this stage
        stage_layers = range(start_layer, end_layer + 1)
        
        if microbatch.is_prefill:
            ops = MacroOperations.estimate_pp_stage_ops(
                self.model_profile,
                stage_layers,
                "prefill",
                microbatch.data_size_tokens,
                self.lod
            )
        else:
            # For decode, batch size is 1
            ops = MacroOperations.estimate_pp_stage_ops(
                self.model_profile,
                stage_layers,
                "decode",
                1,  # Decode processes one token at a time
                self.lod
            )
        
        # Execute computation based on parallelism within stage
        if self.parallelism_strategy == 'TP_PP' and len(self.current_tp_gpu_group) > 1:
            # Distribute across TP GPUs
            tp_degree = len(self.current_tp_gpu_group)
            sharded_ops = {
                'flops_required_fp16': ops['flops_required_fp16'] / tp_degree,
                'memory_read_bytes': ops['memory_read_bytes'] / tp_degree,
                'memory_write_bytes': ops['memory_write_bytes'] / tp_degree,
                'is_memory_bound_hint': not microbatch.is_prefill,
                'task_id': f"{microbatch.request_id}_mb{microbatch.microbatch_idx}_stage{stage_idx}_aggregated"
            }
            
            # Submit parallel tasks
            compute_events = []
            for i, gpu_id in enumerate(self.current_tp_gpu_group):
                task_desc = sharded_ops.copy()
                task_desc['task_id'] = f"{task_desc['task_id']}_tp{i}"
                compute_event = self.virtual_hardware.submit_computation_task(gpu_id, task_desc)
                compute_events.append(compute_event)
            
            # Wait for all shards
            yield simpy.AllOf(self.simpy_env, compute_events)
            
            # Single collective for entire stage
            if tp_degree > 1:
                yield from self._simulate_tp_collective(
                    'AllReduce',
                    self.current_tp_gpu_group,
                    batch_size=1,
                    sequence_length=microbatch.data_size_tokens if microbatch.is_prefill else 1,
                    op_type=f'stage{stage_idx}_{"prefill" if microbatch.is_prefill else "decode"}'
                )
        else:
            # Single GPU computation for entire stage
            gpu_id = self.current_tp_gpu_group[0]
            ops['task_id'] = f"{microbatch.request_id}_mb{microbatch.microbatch_idx}_stage{stage_idx}_aggregated"
            
            yield self.virtual_hardware.submit_computation_task(gpu_id, ops)
    
    def _process_stage_layers_detailed(
        self, stage_idx: int, microbatch: MicrobatchState,
        start_layer: int, end_layer: int
    ) -> simpy.events.Event:
        """Process stage layers individually (high LoD)."""
        # Process each layer in this stage
        for layer_idx in range(start_layer, end_layer + 1):
            # Determine layer type (alternating attention/MLP)
            if layer_idx % 2 == 0:
                layer_type = 'attention'
            else:
                layer_type = 'mlp'
            
            layer_stats = self._get_layer_stats(layer_type)
            if not layer_stats:
                # Fallback stats
                if microbatch.is_prefill:
                    layer_stats = {
                        'flops_per_token_prefill': self.model_profile['prefill_op_stats']['flops_per_token'] / self.model_profile['num_layers'],
                        'memory_bytes_per_token_prefill': self.model_profile['prefill_op_stats']['memory_bytes_per_token'] / self.model_profile['num_layers'],
                        'tp_collective_type': 'AllReduce'
                    }
                else:
                    layer_stats = {
                        'flops_per_token_decode': self.model_profile['decode_op_stats']['flops_per_token'] / self.model_profile['num_layers'],
                        'memory_bytes_per_token_decode': self.model_profile['decode_op_stats']['memory_bytes_per_token'] / self.model_profile['num_layers'],
                        'tp_collective_type': 'AllReduce'
                    }
            
            # Execute layer computation
            if self.parallelism_strategy == 'TP_PP' and len(self.current_tp_gpu_group) > 1:
                # Use tensor parallelism within this stage
                yield from self._dispatch_tp_shardable_operation(
                    layer_stats,
                    batch_size=1,
                    sequence_length=microbatch.data_size_tokens if microbatch.is_prefill else 1,
                    op_type=f'{layer_type}_{"prefill" if microbatch.is_prefill else "decode"}',
                    tp_collective_type=layer_stats.get('tp_collective_type', 'AllReduce'),
                    tp_gpu_group=self.current_tp_gpu_group,
                    request_id=f"{microbatch.request_id}_mb{microbatch.microbatch_idx}_layer{layer_idx}"
                )
            else:
                # Single GPU computation for this stage
                gpu_id = self.current_tp_gpu_group[0]
                
                if microbatch.is_prefill:
                    task_desc = {
                        'flops_required_fp16': layer_stats.get('flops_per_token_prefill', 1e9) * microbatch.data_size_tokens,
                        'memory_read_bytes': layer_stats.get('memory_bytes_per_token_prefill', 1e6) * microbatch.data_size_tokens,
                        'memory_write_bytes': layer_stats.get('memory_bytes_per_token_prefill', 1e6) * microbatch.data_size_tokens / 2,
                        'is_memory_bound_hint': False,
                        'task_id': f"{microbatch.request_id}_mb{microbatch.microbatch_idx}_stage{stage_idx}_layer{layer_idx}"
                    }
                else:
                    task_desc = {
                        'flops_required_fp16': layer_stats.get('flops_per_token_decode', 1e9),
                        'memory_read_bytes': layer_stats.get('memory_bytes_per_token_decode', 1e6),
                        'memory_write_bytes': layer_stats.get('memory_bytes_per_token_decode', 1e6) / 2,
                        'is_memory_bound_hint': True,
                        'task_id': f"{microbatch.request_id}_mb{microbatch.microbatch_idx}_stage{stage_idx}_layer{layer_idx}"
                    }
                
                yield self.virtual_hardware.submit_computation_task(gpu_id, task_desc)
    
    def _transfer_to_next_stage(
        self, stage_idx: int, microbatch: MicrobatchState
    ) -> simpy.events.Event:
        """Transfer activations to the next pipeline stage."""
        # Calculate activation size
        layer_stats = self._get_layer_stats('attention')  # Use attention for activation size
        activation_bytes_per_token = layer_stats.get(
            'activation_output_bytes_per_token',
            self.model_profile.get('hidden_size', 4096) * 2  # FP16
        )
        activation_size = microbatch.data_size_tokens * activation_bytes_per_token
        
        # Determine source and destination GPUs
        source_gpu = self.pp_stage_to_gpus[stage_idx][-1]  # Last GPU of current stage
        dest_gpu = self.pp_stage_to_gpus[stage_idx + 1][0]  # First GPU of next stage
        
        logger.debug(
            f"Transferring {activation_size / 1e6:.2f} MB activations from "
            f"stage {stage_idx} ({source_gpu}) to stage {stage_idx + 1} ({dest_gpu})"
        )
        
        # Submit network transfer
        yield self.virtual_hardware.submit_network_transfer_task(
            source_gpu,
            dest_gpu,
            activation_size
        )
        
        # Update microbatch state and forward to next stage
        microbatch.current_stage = stage_idx + 1
        microbatch.arrival_time_at_stage = self.simpy_env.now
        yield self.stage_input_queues[stage_idx + 1].put(microbatch)
    
    def _complete_microbatch(self, microbatch: MicrobatchState) -> simpy.events.Event:
        """Handle completion of a microbatch at the last stage."""
        # Check if we're tracking this request
        if microbatch.request_id not in self.active_microbatches:
            yield self.simpy_env.timeout(0)  # Still yield control
            return
        
        # Remove this microbatch index from active set
        active_indices = self.active_microbatches[microbatch.request_id]
        active_indices.discard(microbatch.microbatch_idx)
        
        # Check if all microbatches are complete
        if len(active_indices) == 0:
            # All microbatches complete
            del self.active_microbatches[microbatch.request_id]
            
            # Check if sequence is pending deletion
            if microbatch.request_id in self.sequences_pending_deletion:
                # Now safe to remove the sequence
                seq_state = self.sequences_pending_deletion.pop(microbatch.request_id)
                yield from self._release_sequence_resources(seq_state)
                logger.debug(f"Released pending sequence {microbatch.request_id}")
            elif microbatch.is_prefill:
                # Check if ALL prefill microbatches are complete
                if microbatch.request_id in self.prefill_microbatch_count:
                    # This was the last prefill microbatch
                    del self.prefill_microbatch_count[microbatch.request_id]
                    
                    # Prefill complete, update sequence state
                    seq_state = self.running_sequences.get(microbatch.request_id)
                    if seq_state:
                        logger.debug(f"All prefill microbatches complete for {microbatch.request_id}, starting decode")
                        seq_state.status = "DECODING"
                        seq_state.prompt_tokens_processed = seq_state.request.prompt_num_tokens
                        
                        # Log first token time
                        self.metrics_collector.log_first_token_generated(
                            seq_state.request.request_id,
                            self.simpy_env.now,
                            seq_state.prefill_start_time
                        )
                        
                        # Start decode iterations
                        yield from self._start_decode_iterations(seq_state)
                    else:
                        logger.warning(f"No sequence state found for completed prefill {microbatch.request_id}")
        
        yield self.simpy_env.timeout(0)  # Yield control
    
    def _start_decode_iterations(self, seq_state: SequenceState) -> simpy.events.Event:
        """Start decode iterations for a sequence after prefill."""
        request = seq_state.request
        
        # Generate tokens until max_output_tokens
        while seq_state.output_tokens_generated < request.max_output_tokens:
            # Track new decode microbatches
            if request.request_id not in self.active_microbatches:
                self.active_microbatches[request.request_id] = set()
            
            # Create decode microbatches
            for mb_idx in range(self.num_microbatches):
                mb_state = MicrobatchState(
                    request_id=request.request_id,
                    microbatch_idx=mb_idx,
                    current_stage=0,
                    data_size_tokens=1,  # Decode processes one token at a time
                    is_prefill=False,
                    arrival_time_at_stage=self.simpy_env.now
                )
                
                # Track this microbatch
                self.active_microbatches[request.request_id].add(mb_idx)
                
                # Dispatch to first stage
                yield self.stage_input_queues[0].put(mb_state)
            
            # Wait for decode iteration to complete
            # For decode, we need to wait for the token to traverse all stages
            # Each stage processes very quickly for a single token
            # Estimate based on actual compute time + transfer overhead
            
            # Compute time per stage for decode (very small for single token)
            compute_time_per_stage = 0.0001  # 0.1ms per stage for decode
            
            # Transfer time between stages (activation size is small for decode)
            # Hidden size * 2 bytes (fp16) / bandwidth
            hidden_size = self.model_profile.get('hidden_size', 5120)
            activation_bytes = hidden_size * 2  # fp16
            transfer_time = activation_bytes / 600e9  # 600 GB/s NVLink
            
            # Total pipeline latency = compute time + transfers
            total_latency = (compute_time_per_stage * self.pp_stages + 
                           transfer_time * (self.pp_stages - 1))
            
            yield self.simpy_env.timeout(total_latency)
            
            # Update token count
            seq_state.output_tokens_generated += 1
            self.metrics_collector.log_token_decoded(
                request.request_id,
                self.simpy_env.now,
                seq_state.output_tokens_generated
            )
            
            # Check completion
            if seq_state.output_tokens_generated >= request.max_output_tokens:
                self.metrics_collector.log_request_completed(
                    request.request_id,
                    self.simpy_env.now,
                    seq_state.output_tokens_generated,
                    "SUCCESS"
                )
                
                # Clean up - check for active microbatches first
                if request.request_id in self.running_sequences:
                    if request.request_id in self.active_microbatches and len(self.active_microbatches[request.request_id]) > 0:
                        # Defer cleanup
                        self.sequences_pending_deletion[request.request_id] = self.running_sequences.pop(request.request_id)
                        logger.debug(f"Deferring cleanup of completed sequence {request.request_id}")
                    else:
                        # Safe to clean up now
                        self.running_sequences.pop(request.request_id)
                        yield from self._release_sequence_resources(seq_state)
                
                break