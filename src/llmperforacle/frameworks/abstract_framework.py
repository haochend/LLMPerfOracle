"""Abstract base class for LLM serving framework simulations."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

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