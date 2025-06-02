"""
Unit tests for configuration validation system.
"""

import pytest
from llmperforacle.utils.config_validator import (
    ModelConfigValidator,
    HardwareConfigValidator,
    FrameworkConfigValidator,
    ExperimentConfigValidator,
    ConfigurationError
)


class TestModelConfigValidator:
    """Test model configuration validation."""
    
    def test_valid_model_config(self):
        """Test validation of a complete model config."""
        config = {
            'parameters': 7000000000,
            'parameter_bytes_fp16': 14000000000,
            'hidden_size': 4096,
            'num_layers': 32,
            'num_attention_heads': 32,
            'num_kv_heads': 32,
            'intermediate_size': 11008,
            'vocab_size': 32000,
            'prefill_op_stats': {
                'flops_per_token': 14000000000,
                'memory_bytes_per_token': 28000000
            },
            'decode_op_stats': {
                'flops_per_token': 14000000000,
                'memory_bytes_per_token': 56000000
            },
            'kv_cache_bytes_per_token_per_layer': 16384
        }
        
        errors = ModelConfigValidator.validate('TestModel', config)
        assert len(errors) == 0
    
    def test_missing_required_fields(self):
        """Test detection of missing required fields."""
        config = {
            'parameters': 7000000000,
            'hidden_size': 4096,
            # Missing: num_layers, num_attention_heads, etc.
        }
        
        errors = ModelConfigValidator.validate('TestModel', config)
        assert len(errors) > 0
        assert any('missing required fields' in e for e in errors)
    
    def test_auto_add_parameter_bytes(self):
        """Test automatic addition of parameter_bytes_fp16."""
        config = {
            'parameters': 7000000000,
            'hidden_size': 4096,
            'num_layers': 32,
            'num_attention_heads': 32,
            'num_kv_heads': 32,
            'intermediate_size': 11008,
            'vocab_size': 32000,
            'prefill_op_stats': {
                'flops_per_token': 14000000000,
                'memory_bytes_per_token': 28000000
            },
            'decode_op_stats': {
                'flops_per_token': 14000000000,
                'memory_bytes_per_token': 56000000
            }
        }
        
        errors = ModelConfigValidator.validate('TestModel', config)
        # Should auto-add parameter_bytes_fp16
        assert 'parameter_bytes_fp16' in config
        assert config['parameter_bytes_fp16'] == 14000000000
    
    def test_inconsistent_parameter_bytes(self):
        """Test detection of inconsistent parameter_bytes_fp16."""
        config = {
            'parameters': 7000000000,
            'parameter_bytes_fp16': 10000000000,  # Wrong!
            'hidden_size': 4096,
            'num_layers': 32,
            'num_attention_heads': 32,
            'num_kv_heads': 32,
            'intermediate_size': 11008,
            'vocab_size': 32000,
            'prefill_op_stats': {
                'flops_per_token': 14000000000,
                'memory_bytes_per_token': 28000000
            },
            'decode_op_stats': {
                'flops_per_token': 14000000000,
                'memory_bytes_per_token': 56000000
            }
        }
        
        errors = ModelConfigValidator.validate('TestModel', config)
        assert any('Inconsistent parameter_bytes_fp16' in e for e in errors)


class TestHardwareConfigValidator:
    """Test hardware configuration validation."""
    
    def test_valid_hardware_config(self):
        """Test validation of valid hardware config."""
        config = {
            'compute_devices': [
                {
                    'device_id': 'gpu0',
                    'device_type': 'GPU',
                    'peak_tflops': {'fp16': 312, 'int8': 624},
                    'memory_capacity_bytes': 80_000_000_000,
                    'memory_gbps': 2039,
                    'processing_units': 108
                }
            ],
            'network_links': [
                {
                    'link_id': 'client_to_gpu',
                    'source_id': 'client_node_0',
                    'dest_id': 'gpu0',
                    'bandwidth_bps': 10_000_000_000,
                    'latency_s': 0.0001,
                    'bidirectional': True
                }
            ]
        }
        
        errors = HardwareConfigValidator.validate(config)
        assert len(errors) == 0
    
    def test_duplicate_device_ids(self):
        """Test detection of duplicate device IDs."""
        config = {
            'compute_devices': [
                {
                    'device_id': 'gpu0',
                    'device_type': 'GPU',
                    'peak_tflops': {'fp16': 312, 'int8': 624},
                    'memory_capacity_bytes': 80_000_000_000,
                    'memory_gbps': 2039,
                    'processing_units': 108
                },
                {
                    'device_id': 'gpu0',  # Duplicate!
                    'device_type': 'GPU',
                    'peak_tflops': {'fp16': 312, 'int8': 624},
                    'memory_capacity_bytes': 80_000_000_000,
                    'memory_gbps': 2039,
                    'processing_units': 108
                }
            ]
        }
        
        errors = HardwareConfigValidator.validate(config)
        assert any('Duplicate device ID' in e for e in errors)
    
    def test_invalid_memory_capacity(self):
        """Test detection of unrealistic memory capacity."""
        config = {
            'compute_devices': [
                {
                    'device_id': 'gpu0',
                    'device_type': 'GPU',
                    'peak_tflops': {'fp16': 312, 'int8': 624},
                    'memory_capacity_bytes': 500_000_000,  # 0.5 GB - too small!
                    'memory_gbps': 2039,
                    'processing_units': 108
                }
            ]
        }
        
        errors = HardwareConfigValidator.validate(config)
        assert any('Unrealistic memory capacity' in e for e in errors)
    
    def test_invalid_network_link(self):
        """Test detection of invalid network links."""
        config = {
            'compute_devices': [
                {
                    'device_id': 'gpu0',
                    'device_type': 'GPU',
                    'peak_tflops': {'fp16': 312, 'int8': 624},
                    'memory_capacity_bytes': 80_000_000_000,
                    'memory_gbps': 2039,
                    'processing_units': 108
                }
            ],
            'network_links': [
                {
                    'link_id': 'invalid_link',
                    'source_id': 'gpu0',
                    'dest_id': 'gpu1',  # gpu1 doesn't exist!
                    'bandwidth_bps': 10_000_000_000,
                    'latency_s': 0.0001
                }
            ]
        }
        
        errors = HardwareConfigValidator.validate(config)
        assert any('Unknown destination device gpu1' in e for e in errors)


class TestFrameworkConfigValidator:
    """Test framework configuration validation."""
    
    def test_valid_vllm_config(self):
        """Test validation of valid VLLM config."""
        fw_config = {
            'type': 'VLLM',
            'config': {
                'model_profile_id': 'Llama2-7B',
                'gpu_id': 'gpu0',
                'block_size': 16,
                'max_num_seqs': 256
            }
        }
        
        hardware = {
            'compute_devices': [
                {'device_id': 'gpu0', 'device_type': 'GPU'}
            ]
        }
        
        model_db = {
            'Llama2-7B': {'parameters': 7000000000}
        }
        
        errors = FrameworkConfigValidator.validate(fw_config, hardware, model_db)
        assert len(errors) == 0
    
    def test_invalid_gpu_assignment(self):
        """Test detection of invalid GPU assignment."""
        fw_config = {
            'type': 'VLLM',
            'config': {
                'model_profile_id': 'Llama2-7B',
                'gpu_id': 'gpu1',  # Doesn't exist!
                'block_size': 16,
                'max_num_seqs': 256
            }
        }
        
        hardware = {
            'compute_devices': [
                {'device_id': 'gpu0', 'device_type': 'GPU'}
            ]
        }
        
        model_db = {
            'Llama2-7B': {'parameters': 7000000000}
        }
        
        errors = FrameworkConfigValidator.validate(fw_config, hardware, model_db)
        assert any("gpu_id 'gpu1' not found" in e for e in errors)
    
    def test_invalid_model_reference(self):
        """Test detection of invalid model reference."""
        fw_config = {
            'type': 'VLLM',
            'config': {
                'model_profile_id': 'InvalidModel',  # Doesn't exist!
                'gpu_id': 'gpu0',
                'block_size': 16,
                'max_num_seqs': 256
            }
        }
        
        hardware = {
            'compute_devices': [
                {'device_id': 'gpu0', 'device_type': 'GPU'}
            ]
        }
        
        model_db = {
            'Llama2-7B': {'parameters': 7000000000}
        }
        
        errors = FrameworkConfigValidator.validate(fw_config, hardware, model_db)
        assert any('Unknown model profile: InvalidModel' in e for e in errors)
    
    def test_parallel_vllm_validation(self):
        """Test validation of ParallelVLLM config."""
        fw_config = {
            'type': 'ParallelVLLM',
            'config': {
                'model_profile_id': 'Llama2-13B',
                'block_size': 16,
                'max_num_seqs': 256,
                'parallelism': {
                    'strategy': 'TP',
                    'tp_degree': 4,
                    'gpu_ids': ['gpu0', 'gpu1', 'gpu2', 'gpu3']
                }
            }
        }
        
        hardware = {
            'compute_devices': [
                {'device_id': f'gpu{i}', 'device_type': 'GPU'}
                for i in range(4)
            ]
        }
        
        model_db = {
            'Llama2-13B': {'parameters': 13000000000}
        }
        
        errors = FrameworkConfigValidator.validate(fw_config, hardware, model_db)
        assert len(errors) == 0
    
    def test_tp_pp_gpu_count_mismatch(self):
        """Test detection of GPU count mismatch in TP+PP config."""
        fw_config = {
            'type': 'ParallelVLLM',
            'config': {
                'model_profile_id': 'Llama3-70B',
                'block_size': 16,
                'max_num_seqs': 256,
                'parallelism': {
                    'strategy': 'TP_PP',
                    'tp_degree': 4,
                    'pp_stages': 2,
                    'gpu_ids': ['gpu0', 'gpu1', 'gpu2']  # Should be 8!
                }
            }
        }
        
        hardware = {
            'compute_devices': [
                {'device_id': f'gpu{i}', 'device_type': 'GPU'}
                for i in range(8)
            ]
        }
        
        model_db = {
            'Llama3-70B': {'parameters': 70000000000}
        }
        
        errors = FrameworkConfigValidator.validate(fw_config, hardware, model_db)
        assert any('TP_PP expects 8 GPUs' in e for e in errors)


class TestExperimentConfigValidator:
    """Test complete experiment configuration validation."""
    
    def test_complete_valid_config(self):
        """Test validation of a complete valid configuration."""
        config = {
            'simulation': {
                'max_simulation_time': 300,
                'random_seed': 42,
                'lod': 'medium'
            },
            'model_characteristics_db_path': './configs/model_params.json',
            'hardware_profile': {
                'compute_devices': [
                    {
                        'device_id': 'gpu0',
                        'device_type': 'GPU',
                        'peak_tflops': {'fp16': 312, 'int8': 624},
                        'memory_capacity_bytes': 80_000_000_000,
                        'memory_gbps': 2039,
                        'processing_units': 108
                    }
                ],
                'network_links': []
            },
            'workload': {
                'total_duration': 300,
                'client_profiles': [
                    {
                        'inter_arrival_time_dist_config': {'type': 'Exponential', 'rate': 1.0},
                        'prompt_tokens_dist_config': {'type': 'Fixed', 'value': 100},
                        'max_output_tokens_dist_config': {'type': 'Fixed', 'value': 50}
                    }
                ]
            },
            'frameworks_to_test': [
                {
                    'type': 'VLLM',
                    'config': {
                        'model_profile_id': 'Llama2-7B',
                        'gpu_id': 'gpu0',
                        'block_size': 16,
                        'max_num_seqs': 256
                    }
                }
            ],
            'metrics_config': {}
        }
        
        # Note: This will fail if model_characteristics_db_path doesn't exist
        # For unit test, we'll check for that specific error
        is_valid, errors = ExperimentConfigValidator.validate(config)
        
        # Should only have model database path error in unit test environment
        if not is_valid:
            assert len(errors) == 1
            assert 'Model database not found' in errors[0]
    
    def test_invalid_lod_value(self):
        """Test detection of invalid LoD value."""
        config = {
            'simulation': {
                'max_simulation_time': 300,
                'lod': 'ultra'  # Invalid!
            },
            'hardware_profile': {'compute_devices': []},
            'workload': {'total_duration': 300, 'client_profiles': []},
            'frameworks_to_test': []
        }
        
        is_valid, errors = ExperimentConfigValidator.validate(config)
        assert not is_valid
        assert any('Invalid LoD: ultra' in e for e in errors)
    
    def test_negative_simulation_time(self):
        """Test detection of negative simulation time."""
        config = {
            'simulation': {
                'max_simulation_time': -100  # Invalid!
            },
            'hardware_profile': {'compute_devices': []},
            'workload': {'total_duration': 300, 'client_profiles': []},
            'frameworks_to_test': []
        }
        
        is_valid, errors = ExperimentConfigValidator.validate(config)
        assert not is_valid
        assert any('Invalid max_simulation_time: -100' in e for e in errors)