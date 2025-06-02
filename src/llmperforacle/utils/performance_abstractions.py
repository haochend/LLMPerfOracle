"""
Performance modeling abstractions for simulation speedup.

This module provides analytical models and macro operations to reduce
simulation event count while maintaining reasonable accuracy.
"""

import math
from typing import Dict, Any, Optional, Tuple


class CollectiveCommunicationModels:
    """Analytical cost models for collective communication operations."""
    
    @staticmethod
    def simplified_ring_allreduce_time(
        data_size_bytes: float,
        num_gpus: int,
        link_bandwidth_bps: float,
        link_latency_s: float
    ) -> float:
        """
        Estimate time for ring all-reduce operation.
        
        Based on the alpha-beta model: latency term (alpha) + bandwidth term (beta * size)
        For ring all-reduce: 2 * (N-1)/N * (data_size / bandwidth) + 2 * (N-1) * latency
        
        Args:
            data_size_bytes: Size of data to reduce in bytes
            num_gpus: Number of GPUs in the ring
            link_bandwidth_bps: Bandwidth of inter-GPU links in bits per second
            link_latency_s: Latency of inter-GPU links in seconds
            
        Returns:
            Estimated time in seconds
        """
        if num_gpus <= 1:
            return 0.0
            
        # Convert bandwidth from bits/s to bytes/s
        link_bandwidth_Bps = link_bandwidth_bps / 8
        
        # Ring algorithm: data is chunked and sent around the ring
        # Each GPU sends/receives (N-1)/N of the data
        # Total data movement: 2 * (N-1)/N * data_size (reduce + broadcast phases)
        data_per_step = data_size_bytes * (num_gpus - 1) / num_gpus
        
        # Bandwidth term: time to transfer data
        bandwidth_time = 2 * data_per_step / link_bandwidth_Bps
        
        # Latency term: 2 * (N-1) steps for reduce + broadcast
        latency_time = 2 * (num_gpus - 1) * link_latency_s
        
        return bandwidth_time + latency_time
    
    @staticmethod
    def simplified_allgather_time(
        data_size_bytes: float,
        num_gpus: int,
        link_bandwidth_bps: float,
        link_latency_s: float
    ) -> float:
        """
        Estimate time for all-gather operation.
        
        In all-gather, each GPU starts with a piece of data and ends with all pieces.
        
        Args:
            data_size_bytes: Size of data per GPU in bytes
            num_gpus: Number of GPUs
            link_bandwidth_bps: Bandwidth of inter-GPU links in bits per second
            link_latency_s: Latency of inter-GPU links in seconds
            
        Returns:
            Estimated time in seconds
        """
        if num_gpus <= 1:
            return 0.0
            
        # Convert bandwidth from bits/s to bytes/s
        link_bandwidth_Bps = link_bandwidth_bps / 8
        
        # Ring algorithm: (N-1) steps, each GPU sends its chunk
        # Total data received by each GPU: (N-1) * data_size_bytes
        total_data_transfer = (num_gpus - 1) * data_size_bytes
        
        # Bandwidth term
        bandwidth_time = total_data_transfer / link_bandwidth_Bps
        
        # Latency term: (N-1) steps
        latency_time = (num_gpus - 1) * link_latency_s
        
        return bandwidth_time + latency_time
    
    @staticmethod
    def simplified_reduce_scatter_time(
        data_size_bytes: float,
        num_gpus: int,
        link_bandwidth_bps: float,
        link_latency_s: float
    ) -> float:
        """
        Estimate time for reduce-scatter operation.
        
        Each GPU ends with 1/N of the reduced data.
        
        Args:
            data_size_bytes: Total size of data to reduce in bytes
            num_gpus: Number of GPUs
            link_bandwidth_bps: Bandwidth of inter-GPU links in bits per second
            link_latency_s: Latency of inter-GPU links in seconds
            
        Returns:
            Estimated time in seconds
        """
        if num_gpus <= 1:
            return 0.0
            
        # Convert bandwidth from bits/s to bytes/s
        link_bandwidth_Bps = link_bandwidth_bps / 8
        
        # Ring algorithm: each GPU sends (N-1)/N of its data
        data_per_gpu = data_size_bytes * (num_gpus - 1) / num_gpus
        
        # Bandwidth term
        bandwidth_time = data_per_gpu / link_bandwidth_Bps
        
        # Latency term: (N-1) steps
        latency_time = (num_gpus - 1) * link_latency_s
        
        return bandwidth_time + latency_time
    
    @staticmethod
    def get_collective_time(
        collective_type: str,
        data_size_bytes: float,
        num_gpus: int,
        link_bandwidth_bps: float,
        link_latency_s: float
    ) -> float:
        """
        Get estimated time for a collective operation.
        
        Args:
            collective_type: Type of collective ("AllReduce", "AllGather", "ReduceScatter")
            data_size_bytes: Size of data in bytes
            num_gpus: Number of GPUs
            link_bandwidth_bps: Bandwidth of inter-GPU links in bits per second
            link_latency_s: Latency of inter-GPU links in seconds
            
        Returns:
            Estimated time in seconds
        """
        if collective_type == "AllReduce":
            return CollectiveCommunicationModels.simplified_ring_allreduce_time(
                data_size_bytes, num_gpus, link_bandwidth_bps, link_latency_s
            )
        elif collective_type == "AllGather":
            return CollectiveCommunicationModels.simplified_allgather_time(
                data_size_bytes, num_gpus, link_bandwidth_bps, link_latency_s
            )
        elif collective_type == "ReduceScatter":
            return CollectiveCommunicationModels.simplified_reduce_scatter_time(
                data_size_bytes, num_gpus, link_bandwidth_bps, link_latency_s
            )
        else:
            raise ValueError(f"Unknown collective type: {collective_type}")


class MacroOperations:
    """Macro operation calculators for aggregated computation."""
    
    @staticmethod
    def estimate_macro_prefill_ops(
        model_profile: Dict[str, Any],
        num_tokens_to_prefill: int,
        lod: str = "medium"
    ) -> Dict[str, float]:
        """
        Estimate aggregated operations for entire prefill phase.
        
        Args:
            model_profile: Model characteristics including aggregated_ops
            num_tokens_to_prefill: Number of tokens to prefill
            lod: Level of detail ("high" for per-layer, "medium" for aggregated)
            
        Returns:
            Dictionary with flops_required_fp16, memory_read_bytes, memory_write_bytes
        """
        if lod == "medium" and "aggregated_ops" in model_profile:
            # Use pre-calculated aggregated stats
            agg_stats = model_profile["aggregated_ops"]["prefill"]
            total_flops = agg_stats["total_flops_per_prompt_token"] * num_tokens_to_prefill
            total_memory_bytes = agg_stats["total_memory_bytes_per_prompt_token"] * num_tokens_to_prefill
            
            return {
                "flops_required_fp16": total_flops,
                "memory_read_bytes": total_memory_bytes,
                "memory_write_bytes": total_memory_bytes / 2  # Assume 2:1 read:write ratio
            }
        else:
            # Fall back to detailed calculation
            prefill_stats = model_profile.get("prefill_op_stats", {})
            total_flops = prefill_stats.get("flops_per_token", 0) * num_tokens_to_prefill
            total_memory_bytes = prefill_stats.get("memory_bytes_per_token", 0) * num_tokens_to_prefill
            
            return {
                "flops_required_fp16": total_flops,
                "memory_read_bytes": total_memory_bytes,
                "memory_write_bytes": total_memory_bytes / 2
            }
    
    @staticmethod
    def estimate_macro_decode_ops(
        model_profile: Dict[str, Any],
        current_batch_size: int,
        lod: str = "medium"
    ) -> Dict[str, float]:
        """
        Estimate aggregated operations for decode step.
        
        Args:
            model_profile: Model characteristics including aggregated_ops
            current_batch_size: Number of sequences in batch
            lod: Level of detail ("high" for per-layer, "medium" for aggregated)
            
        Returns:
            Dictionary with flops_required_fp16, memory_read_bytes, memory_write_bytes
        """
        if lod == "medium" and "aggregated_ops" in model_profile:
            # Use pre-calculated aggregated stats
            agg_stats = model_profile["aggregated_ops"]["decode"]
            total_flops = agg_stats["total_flops_per_token_in_batch"] * current_batch_size
            total_memory_bytes = agg_stats["total_memory_bytes_per_token_in_batch"] * current_batch_size
            
            return {
                "flops_required_fp16": total_flops,
                "memory_read_bytes": total_memory_bytes,
                "memory_write_bytes": total_memory_bytes / 4  # Decode is more read-heavy
            }
        else:
            # Fall back to detailed calculation
            decode_stats = model_profile.get("decode_op_stats", {})
            total_flops = decode_stats.get("flops_per_token", 0) * current_batch_size
            total_memory_bytes = decode_stats.get("memory_bytes_per_token", 0) * current_batch_size
            
            return {
                "flops_required_fp16": total_flops,
                "memory_read_bytes": total_memory_bytes,
                "memory_write_bytes": total_memory_bytes / 4
            }
    
    @staticmethod
    def estimate_pp_stage_ops(
        model_profile: Dict[str, Any],
        stage_layers: range,
        operation_type: str,
        num_tokens: int,
        lod: str = "medium"
    ) -> Dict[str, float]:
        """
        Estimate operations for a pipeline parallel stage.
        
        Args:
            model_profile: Model characteristics
            stage_layers: Range of layer indices for this stage
            operation_type: "prefill" or "decode"
            num_tokens: Number of tokens to process
            lod: Level of detail
            
        Returns:
            Dictionary with flops_required_fp16, memory_read_bytes, memory_write_bytes
        """
        num_layers_in_stage = len(stage_layers)
        total_layers = model_profile["num_layers"]
        
        if lod == "medium" and "aggregated_ops" in model_profile:
            # Scale aggregated ops by fraction of layers
            layer_fraction = num_layers_in_stage / total_layers
            
            if operation_type == "prefill":
                full_ops = MacroOperations.estimate_macro_prefill_ops(model_profile, num_tokens, lod)
            else:
                full_ops = MacroOperations.estimate_macro_decode_ops(model_profile, num_tokens, lod)
            
            return {
                "flops_required_fp16": full_ops["flops_required_fp16"] * layer_fraction,
                "memory_read_bytes": full_ops["memory_read_bytes"] * layer_fraction,
                "memory_write_bytes": full_ops["memory_write_bytes"] * layer_fraction
            }
        else:
            # Detailed per-layer calculation
            total_flops = 0
            total_memory_read = 0
            total_memory_write = 0
            
            for layer_type in ["attention", "mlp"]:
                layer_stats = model_profile["layer_types"][layer_type]
                
                if operation_type == "prefill":
                    flops_key = "flops_per_token_prefill"
                    memory_key = "memory_bytes_per_token_prefill"
                else:
                    flops_key = "flops_per_token_decode"
                    memory_key = "memory_bytes_per_token_decode"
                
                # Each layer has attention + MLP
                flops_per_layer = layer_stats[flops_key] * 2  # attention + mlp
                memory_per_layer = layer_stats[memory_key] * 2
                
                total_flops += flops_per_layer * num_layers_in_stage * num_tokens
                total_memory_read += memory_per_layer * num_layers_in_stage * num_tokens
                total_memory_write += memory_per_layer * num_layers_in_stage * num_tokens / 2
            
            return {
                "flops_required_fp16": total_flops,
                "memory_read_bytes": total_memory_read,
                "memory_write_bytes": total_memory_write
            }


def get_simulation_lod(config: Dict[str, Any]) -> str:
    """
    Get the Level of Detail setting from configuration.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Level of detail string ("high" or "medium")
    """
    # Check simulation config
    simulation_config = config.get("simulation", {})
    lod = simulation_config.get("lod", "high")
    
    # Validate
    if lod not in ["high", "medium"]:
        raise ValueError(f"Invalid LoD setting: {lod}. Must be 'high' or 'medium'")
    
    return lod