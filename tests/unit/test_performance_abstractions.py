"""
Unit tests for performance abstraction modules.
"""

import pytest
from llmperforacle.utils.performance_abstractions import (
    CollectiveCommunicationModels,
    MacroOperations,
    get_simulation_lod
)


class TestCollectiveCommunicationModels:
    """Test analytical collective communication models."""
    
    def test_allreduce_single_gpu(self):
        """Test that single GPU returns 0 time."""
        time = CollectiveCommunicationModels.simplified_ring_allreduce_time(
            data_size_bytes=1024,
            num_gpus=1,
            link_bandwidth_bps=100e9,
            link_latency_s=1e-6
        )
        assert time == 0.0
    
    def test_allreduce_multiple_gpus(self):
        """Test ring allreduce with multiple GPUs."""
        time = CollectiveCommunicationModels.simplified_ring_allreduce_time(
            data_size_bytes=1024 * 1024,  # 1MB
            num_gpus=4,
            link_bandwidth_bps=100e9,  # 100 Gbps
            link_latency_s=1e-6  # 1 microsecond
        )
        # Should be positive and reasonable
        assert time > 0
        assert time < 1.0  # Should be less than 1 second for 1MB on fast link
    
    def test_allgather(self):
        """Test allgather collective."""
        time = CollectiveCommunicationModels.simplified_allgather_time(
            data_size_bytes=1024 * 1024,  # 1MB per GPU
            num_gpus=4,
            link_bandwidth_bps=100e9,
            link_latency_s=1e-6
        )
        assert time > 0
        # Allgather should take more time than allreduce for same data size
        # because it transfers (N-1) * data_size total
    
    def test_reduce_scatter(self):
        """Test reduce-scatter collective."""
        time = CollectiveCommunicationModels.simplified_reduce_scatter_time(
            data_size_bytes=4 * 1024 * 1024,  # 4MB total
            num_gpus=4,
            link_bandwidth_bps=100e9,
            link_latency_s=1e-6
        )
        assert time > 0
    
    def test_get_collective_time(self):
        """Test the general collective time function."""
        # Test AllReduce
        time = CollectiveCommunicationModels.get_collective_time(
            "AllReduce",
            data_size_bytes=1024 * 1024,
            num_gpus=4,
            link_bandwidth_bps=100e9,
            link_latency_s=1e-6
        )
        assert time > 0
        
        # Test unknown collective
        with pytest.raises(ValueError):
            CollectiveCommunicationModels.get_collective_time(
                "UnknownCollective",
                data_size_bytes=1024,
                num_gpus=4,
                link_bandwidth_bps=100e9,
                link_latency_s=1e-6
            )


class TestMacroOperations:
    """Test macro operation calculators."""
    
    @pytest.fixture
    def sample_model_profile(self):
        """Sample model profile with aggregated ops."""
        return {
            'parameters': 7000000000,
            'hidden_size': 4096,
            'num_layers': 32,
            'prefill_op_stats': {
                'flops_per_token': 14000000000,
                'memory_bytes_per_token': 28000000
            },
            'decode_op_stats': {
                'flops_per_token': 14000000000,
                'memory_bytes_per_token': 56000000
            },
            'aggregated_ops': {
                'prefill': {
                    'total_flops_per_prompt_token': 14000000000,
                    'total_memory_bytes_per_prompt_token': 28000000,
                    'critical_path_factor': 1.0
                },
                'decode': {
                    'total_flops_per_token_in_batch': 14000000000,
                    'total_memory_bytes_per_token_in_batch': 56000000,
                    'critical_path_factor': 1.0
                }
            }
        }
    
    def test_macro_prefill_ops_medium_lod(self, sample_model_profile):
        """Test macro prefill operations with medium LoD."""
        ops = MacroOperations.estimate_macro_prefill_ops(
            sample_model_profile,
            num_tokens_to_prefill=100,
            lod="medium"
        )
        
        assert 'flops_required_fp16' in ops
        assert 'memory_read_bytes' in ops
        assert 'memory_write_bytes' in ops
        
        # Check values match expectations
        assert ops['flops_required_fp16'] == 14000000000 * 100
        assert ops['memory_read_bytes'] == 28000000 * 100
        assert ops['memory_write_bytes'] == 28000000 * 100 / 2
    
    def test_macro_prefill_ops_high_lod(self, sample_model_profile):
        """Test macro prefill operations with high LoD (fallback)."""
        ops = MacroOperations.estimate_macro_prefill_ops(
            sample_model_profile,
            num_tokens_to_prefill=100,
            lod="high"
        )
        
        # Should use regular prefill_op_stats
        assert ops['flops_required_fp16'] == 14000000000 * 100
    
    def test_macro_decode_ops(self, sample_model_profile):
        """Test macro decode operations."""
        batch_size = 8
        ops = MacroOperations.estimate_macro_decode_ops(
            sample_model_profile,
            current_batch_size=batch_size,
            lod="medium"
        )
        
        assert ops['flops_required_fp16'] == 14000000000 * batch_size
        assert ops['memory_read_bytes'] == 56000000 * batch_size
        assert ops['memory_write_bytes'] == 56000000 * batch_size / 4
    
    def test_pp_stage_ops(self, sample_model_profile):
        """Test pipeline parallel stage operations."""
        # Test with 2 stages, each gets half the layers
        stage_layers = range(0, 16)  # First 16 of 32 layers
        
        ops = MacroOperations.estimate_pp_stage_ops(
            sample_model_profile,
            stage_layers=stage_layers,
            operation_type="prefill",
            num_tokens=100,
            lod="medium"
        )
        
        # Should be half of the full model ops
        full_ops = MacroOperations.estimate_macro_prefill_ops(
            sample_model_profile, 100, "medium"
        )
        
        assert ops['flops_required_fp16'] == pytest.approx(full_ops['flops_required_fp16'] / 2)
        assert ops['memory_read_bytes'] == pytest.approx(full_ops['memory_read_bytes'] / 2)


class TestGetSimulationLoD:
    """Test LoD configuration parsing."""
    
    def test_get_lod_from_simulation_config(self):
        """Test getting LoD from simulation config."""
        config = {
            'simulation': {
                'lod': 'medium'
            }
        }
        assert get_simulation_lod(config) == 'medium'
    
    def test_get_lod_default(self):
        """Test default LoD when not specified."""
        config = {
            'simulation': {}
        }
        assert get_simulation_lod(config) == 'high'
    
    def test_get_lod_invalid(self):
        """Test invalid LoD value."""
        config = {
            'simulation': {
                'lod': 'invalid'
            }
        }
        with pytest.raises(ValueError):
            get_simulation_lod(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])