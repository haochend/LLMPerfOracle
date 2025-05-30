"""Abstract base class for LLM serving framework simulations."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

import simpy

logger = logging.getLogger(__name__)


class AbstractLLMFramework(ABC):
    """Abstract base class defining the interface for LLM framework simulations.
    
    All concrete LLM framework implementations must inherit from this class
    and implement the abstract methods.
    """
    
    def __init__(
        self,
        framework_id: str,
        simpy_env: simpy.Environment,
        framework_specific_config: Dict[str, Any],
        virtual_hardware: Any,  # Will be VirtualHardwarePlatform
        metrics_collector: Any,  # Will be MetricsCollector
        model_profile: Dict[str, Any],
    ):
        """Initialize the framework simulation.
        
        Args:
            framework_id: Unique identifier for this framework instance
            simpy_env: SimPy environment for discrete event simulation
            framework_specific_config: Configuration specific to this framework
            virtual_hardware: Virtual hardware platform instance
            metrics_collector: Metrics collection instance
            model_profile: Model characteristics (from model database)
        """
        self.framework_id = framework_id
        self.simpy_env = simpy_env
        self.config = framework_specific_config
        self.virtual_hardware = virtual_hardware
        self.metrics_collector = metrics_collector
        self.model_profile = model_profile
        
        # Common request queue for incoming requests
        self.request_arrival_queue = simpy.Store(simpy_env)
        
        # Parse parallelism configuration
        self.parallelism_config = self.config.get('parallelism', {})
        self._parse_parallelism_config()
        
        logger.info(f"Initialized {self.__class__.__name__} framework: {framework_id}")
    
    @abstractmethod
    def handle_incoming_request(self, request: Any) -> simpy.Process:
        """Handle a new incoming request.
        
        This method is called by the workload generator when a request
        arrives for this framework.
        
        Args:
            request: Request object from the workload generator
            
        Returns:
            SimPy process handling the request arrival
        """
        pass
    
    @abstractmethod
    def processing_loop(self) -> simpy.events.Event:
        """Main processing loop for the framework.
        
        This method contains the core logic for:
        - Request scheduling
        - Batch formation
        - KV cache management
        - Interaction with virtual hardware
        - Response generation
        
        Returns:
            SimPy event/process for the main loop
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the framework.
        
        Returns:
            Dictionary containing framework state information
        """
        pass
    
    def _estimate_prefill_ops(
        self, num_prompt_tokens: int, batch_size: int = 1
    ) -> Dict[str, Any]:
        """Estimate computational requirements for prefill operation.
        
        Args:
            num_prompt_tokens: Number of tokens in the prompt
            batch_size: Number of sequences in the batch
            
        Returns:
            Dictionary with flops, memory_read_bytes, memory_write_bytes
        """
        prefill_stats = self.model_profile.get("prefill_op_stats", {})
        flops_per_token = prefill_stats.get("flops_per_token", 2e9)
        memory_bytes_per_token = prefill_stats.get("memory_bytes_per_token", 1e3)
        
        return {
            "flops_required_fp16": flops_per_token * num_prompt_tokens * batch_size,
            "memory_read_bytes": memory_bytes_per_token * num_prompt_tokens * batch_size,
            "memory_write_bytes": memory_bytes_per_token * num_prompt_tokens * batch_size / 2,
            "is_memory_bound_hint": False,  # Prefill is typically compute-bound
        }
    
    def _estimate_decode_op(self, batch_size: int) -> Dict[str, Any]:
        """Estimate computational requirements for one decode step.
        
        Args:
            batch_size: Number of sequences in the batch
            
        Returns:
            Dictionary with flops, memory_read_bytes, memory_write_bytes
        """
        decode_stats = self.model_profile.get("decode_op_stats", {})
        flops_per_token = decode_stats.get("flops_per_token", 3e9)
        memory_bytes_per_token = decode_stats.get("memory_bytes_per_token", 1.5e3)
        
        return {
            "flops_required_fp16": flops_per_token * batch_size,
            "memory_read_bytes": memory_bytes_per_token * batch_size,
            "memory_write_bytes": memory_bytes_per_token * batch_size / 2,
            "is_memory_bound_hint": True,  # Decode is typically memory-bound
        }
    
    def _estimate_kv_cache_request_bytes(self, num_tokens: int) -> int:
        """Estimate KV cache size for a given number of tokens.
        
        Args:
            num_tokens: Number of tokens
            
        Returns:
            Estimated size in bytes
        """
        bytes_per_token_per_layer = self.model_profile.get(
            "kv_cache_bytes_per_token_per_layer", 2 * 4096 * 2  # K and V, hidden_size, FP16
        )
        num_layers = self.model_profile.get("num_layers", 32)
        
        return num_tokens * bytes_per_token_per_layer * num_layers
    
    def _parse_parallelism_config(self):
        """Parse and validate parallelism configuration."""
        # Default: single GPU, no parallelism
        if not self.parallelism_config:
            # Try to get gpu_id from top-level config for backward compatibility
            gpu_id = self.config.get('gpu_id', 'gpu0')
            self.parallelism_config = {
                'strategy': 'None',
                'gpu_ids': [gpu_id]
            }
        
        self.parallelism_strategy = self.parallelism_config.get('strategy', 'None')
        self.gpu_ids = self.parallelism_config.get('gpu_ids', [])
        
        # Tensor Parallelism configuration
        self.tp_degree = self.parallelism_config.get('tp_degree', 1)
        
        # Pipeline Parallelism configuration
        self.pp_stages = self.parallelism_config.get('pp_stages', 1)
        self.num_microbatches = self.parallelism_config.get('num_microbatches_per_request', 1)
        
        # Validate configuration
        if self.parallelism_strategy == 'TP':
            if len(self.gpu_ids) != self.tp_degree:
                raise ValueError(f"TP requires {self.tp_degree} GPUs but got {len(self.gpu_ids)}")
        elif self.parallelism_strategy == 'PP':
            if len(self.gpu_ids) % self.pp_stages != 0:
                raise ValueError(f"PP with {self.pp_stages} stages requires GPU count divisible by {self.pp_stages}")
        elif self.parallelism_strategy == 'TP_PP':
            expected_gpus = self.tp_degree * self.pp_stages
            if len(self.gpu_ids) != expected_gpus:
                raise ValueError(f"TP_PP requires {expected_gpus} GPUs (tp={self.tp_degree} * pp={self.pp_stages}) but got {len(self.gpu_ids)}")
        
        # Set up GPU mappings
        self._setup_gpu_mappings()
        
        logger.info(f"Parallelism config: strategy={self.parallelism_strategy}, tp_degree={self.tp_degree}, pp_stages={self.pp_stages}, gpus={self.gpu_ids}")
    
    def _setup_gpu_mappings(self):
        """Set up GPU mappings for different parallelism strategies."""
        if self.parallelism_strategy == 'None':
            self.primary_gpu_id = self.gpu_ids[0] if self.gpu_ids else 'gpu0'
            self.tp_gpu_groups = [[self.primary_gpu_id]]
            self.pp_stage_to_gpus = {0: [self.primary_gpu_id]}
            
        elif self.parallelism_strategy == 'TP':
            self.tp_gpu_groups = [self.gpu_ids]
            self.pp_stage_to_gpus = {0: self.gpu_ids}
            
        elif self.parallelism_strategy == 'PP':
            gpus_per_stage = len(self.gpu_ids) // self.pp_stages
            self.pp_stage_to_gpus = {}
            for stage in range(self.pp_stages):
                start_idx = stage * gpus_per_stage
                end_idx = start_idx + gpus_per_stage
                self.pp_stage_to_gpus[stage] = self.gpu_ids[start_idx:end_idx]
            self.tp_gpu_groups = list(self.pp_stage_to_gpus.values())
            
        elif self.parallelism_strategy == 'TP_PP':
            # GPUs are arranged as pp_stages x tp_degree
            self.pp_stage_to_gpus = {}
            self.tp_gpu_groups = []
            for stage in range(self.pp_stages):
                start_idx = stage * self.tp_degree
                end_idx = start_idx + self.tp_degree
                stage_gpus = self.gpu_ids[start_idx:end_idx]
                self.pp_stage_to_gpus[stage] = stage_gpus
                self.tp_gpu_groups.append(stage_gpus)
        
        # Calculate layer distribution for PP
        if self.pp_stages > 1:
            num_layers = self.model_profile.get('num_layers', 32)
            layers_per_stage = num_layers // self.pp_stages
            self.stage_layer_ranges = {}
            for stage in range(self.pp_stages):
                start_layer = stage * layers_per_stage
                end_layer = start_layer + layers_per_stage - 1
                if stage == self.pp_stages - 1:  # Last stage gets remaining layers
                    end_layer = num_layers - 1
                self.stage_layer_ranges[stage] = (start_layer, end_layer)
        else:
            self.stage_layer_ranges = {0: (0, self.model_profile.get('num_layers', 32) - 1)}
    
    def _get_layer_stats(self, layer_type: str) -> Dict[str, Any]:
        """Get computational stats for a specific layer type.
        
        Args:
            layer_type: Type of layer ('attention' or 'mlp')
            
        Returns:
            Dictionary with layer statistics
        """
        layer_types = self.model_profile.get('layer_types', {})
        return layer_types.get(layer_type, {})
    
    def _estimate_collective_time(self, data_size_bytes: int, collective_type: str, gpu_group: List[str]) -> float:
        """Estimate time for collective communication operation.
        
        Args:
            data_size_bytes: Size of data to communicate
            collective_type: Type of collective ('AllReduce', 'AllGather', etc.)
            gpu_group: List of GPU IDs participating in the collective
            
        Returns:
            Estimated time in seconds
        """
        # Simplified model: assume ring allreduce
        # Time = 2 * (N-1) / N * data_size / bandwidth
        num_gpus = len(gpu_group)
        if num_gpus <= 1:
            return 0.0
        
        # Find slowest link bandwidth in the group (simplified)
        # In reality, would need to analyze the actual topology
        min_bandwidth = float('inf')
        for i in range(num_gpus):
            for j in range(i + 1, num_gpus):
                # This is a simplification - actual implementation would check real links
                min_bandwidth = min(min_bandwidth, 600e9)  # Assume NVLink bandwidth
        
        if collective_type == 'AllReduce':
            # Ring allreduce: 2*(N-1)/N * data_size / bandwidth
            collective_factor = 2 * (num_gpus - 1) / num_gpus
        elif collective_type == 'AllGather':
            # All-gather: (N-1)/N * data_size / bandwidth
            collective_factor = (num_gpus - 1) / num_gpus
        elif collective_type == 'ReduceScatter':
            # Reduce-scatter: (N-1)/N * data_size / bandwidth
            collective_factor = (num_gpus - 1) / num_gpus
        else:
            collective_factor = 1.0
        
        bandwidth_bytes_per_sec = min_bandwidth / 8  # Convert bits to bytes
        return collective_factor * data_size_bytes / bandwidth_bytes_per_sec