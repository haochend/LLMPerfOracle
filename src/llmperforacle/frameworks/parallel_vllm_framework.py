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
        
        # Start pipeline stage worker processes
        for stage_idx in range(self.pp_stages):
            self.simpy_env.process(self._pipeline_stage_worker_process(stage_idx))
        
        logger.info(
            f"Initialized pipeline parallelism with {self.pp_stages} stages"
        )
    
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
        logger.info(f"Starting PP processing loop for {self.framework_id}")
        
        while True:
            # Try to admit new sequences from arrival queue
            newly_admitted = yield from self._admit_new_sequences()
            
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
        
        # Track microbatches for this request
        self.active_microbatches[request.request_id] = microbatch_states
        
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
        
        # Get sequence state
        seq_state = self.running_sequences.get(microbatch.request_id)
        if not seq_state:
            logger.warning(f"Sequence {microbatch.request_id} not found")
            return
        
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
        # Check if all microbatches for this request are complete
        if microbatch.request_id not in self.active_microbatches:
            return
        
        microbatch_states = self.active_microbatches[microbatch.request_id]
        microbatch_states[microbatch.microbatch_idx].current_stage = -1  # Mark as complete
        
        # Check if all microbatches are complete
        all_complete = all(mb.current_stage == -1 for mb in microbatch_states)
        
        if all_complete and microbatch.is_prefill:
            # Prefill complete, start decode iterations
            seq_state = self.running_sequences.get(microbatch.request_id)
            if seq_state:
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
        
        yield self.simpy_env.timeout(0)  # Yield control
    
    def _start_decode_iterations(self, seq_state: SequenceState) -> simpy.events.Event:
        """Start decode iterations for a sequence after prefill."""
        request = seq_state.request
        
        # Generate tokens until max_output_tokens
        while seq_state.output_tokens_generated < request.max_output_tokens:
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
                
                # Dispatch to first stage
                yield self.stage_input_queues[0].put(mb_state)
            
            # Wait for decode iteration to complete (simplified)
            # In reality, would need proper synchronization
            yield self.simpy_env.timeout(0.001)
            
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
                
                # Clean up
                if request.request_id in self.running_sequences:
                    self.running_sequences.pop(request.request_id)
                    yield from self._release_sequence_resources(seq_state)
                
                break