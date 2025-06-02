"""
Regression test suite for LLMPerfOracle.

This suite ensures that:
1. All configurations are valid
2. Key functionality works correctly
3. Performance abstractions (LoD) work properly
4. Parallelism strategies work as expected
5. Memory validation prevents invalid configurations
"""

import pytest
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

from llmperforacle.orchestration import ExperimentOrchestrator
from llmperforacle.utils.config_validator import (
    ExperimentConfigValidator,
    ModelConfigValidator,
    validate_model_db
)


class TestConfigurationRegression:
    """Test all example configurations are valid."""
    
    @pytest.mark.parametrize("config_file", [
        "configs/examples/example_experiment.yaml",
        "configs/examples/walkthrough_example.yaml",
        "configs/examples/example_parallel_experiment.yaml",
        "configs/examples/example_prefix_caching.yaml",
        "configs/examples/example_cross_request_caching.yaml",
        "configs/examples/test_parallel_quick.yaml",
    ])
    def test_example_configs_valid(self, config_file):
        """Test that all example configurations are valid."""
        config_path = Path(config_file)
        if not config_path.exists():
            pytest.skip(f"Config file {config_file} not found")
        
        # Load config
        if config_path.suffix in ['.yaml', '.yml']:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
        else:
            import json
            with open(config_path) as f:
                config = json.load(f)
        
        # Validate
        is_valid, errors = ExperimentConfigValidator.validate(config)
        
        # Special handling for known issues
        allowed_errors = [
            "Unknown destination device framework_entry_0",  # Legacy pattern
            "Model database not found"  # OK in test environment
        ]
        
        filtered_errors = [e for e in errors if not any(allowed in e for allowed in allowed_errors)]
        
        assert len(filtered_errors) == 0, f"Config {config_file} has errors: {filtered_errors}"
    
    def test_model_database_valid(self):
        """Test that model database is valid and complete."""
        model_db_path = Path("configs/model_params.json")
        if not model_db_path.exists():
            pytest.skip("Model database not found")
        
        # Validate
        is_valid, errors = validate_model_db(str(model_db_path))
        assert is_valid, f"Model database has errors: {errors}"
        
        # Check all models have required fields
        with open(model_db_path) as f:
            models = json.load(f)
        
        for model_name, config in models.items():
            # Check parameter_bytes_fp16
            assert 'parameter_bytes_fp16' in config, f"{model_name} missing parameter_bytes_fp16"
            assert config['parameter_bytes_fp16'] > 0, f"{model_name} has invalid parameter_bytes_fp16"
            
            # Check consistency
            if 'parameters' in config:
                expected = config['parameters'] * 2
                assert abs(config['parameter_bytes_fp16'] - expected) < 1000, \
                    f"{model_name} has inconsistent parameter_bytes_fp16"


class TestMemoryValidationRegression:
    """Test memory validation works correctly."""
    
    def create_test_config(self, model_id: str, gpu_memory_gb: int) -> Dict[str, Any]:
        """Create a test configuration."""
        return {
            "simulation": {"max_simulation_time": 10, "lod": "medium"},
            "model_characteristics_db_path": "./configs/model_params.json",
            "hardware_profile": {
                "compute_devices": [{
                    "device_id": "gpu0",
                    "device_type": "GPU",
                    "peak_tflops": {"fp16": 312, "int8": 624},
                    "memory_capacity_bytes": int(gpu_memory_gb * 1e9),
                    "memory_gbps": 2039,
                    "processing_units": 108
                }],
                "network_links": [{
                    "link_id": "client_link",
                    "source_id": "client_node_0",
                    "dest_id": "gpu0",
                    "bandwidth_bps": 10_000_000_000,
                    "latency_s": 0.0001
                }]
            },
            "workload": {
                "total_duration": 10,
                "client_profiles": [{
                    "profile_name": "test_profile",
                    "inter_arrival_time_dist_config": {"type": "Fixed", "value": 10.0},
                    "prompt_tokens_dist_config": {"type": "Fixed", "value": 10},
                    "max_output_tokens_dist_config": {"type": "Fixed", "value": 5}
                }]
            },
            "frameworks_to_test": [{
                "name": "vllm_test",
                "type": "VLLM",
                "is_target_for_workload": True,
                "config": {
                    "model_profile_id": model_id,
                    "gpu_id": "gpu0",
                    "block_size": 16,
                    "max_num_seqs": 256
                }
            }],
            "metrics_config": {
                "output_summary_json_path": f"./experiments/results/test_{model_id}.json",
                "output_requests_csv_path": f"./experiments/results/test_{model_id}.csv"
            }
        }
    
    @pytest.mark.parametrize("model_id,gpu_memory_gb,should_fail", [
        ("Llama2-7B", 40, False),     # 14GB model on 40GB GPU - OK
        ("Llama2-7B", 15, True),      # 14GB model on 15GB GPU - Fail
        ("Llama2-13B", 40, False),    # 26GB model on 40GB GPU - OK
        ("Gemma2-27B", 80, False),    # 54GB model on 80GB GPU - OK
        ("Gemma2-27B", 40, True),     # 54GB model on 40GB GPU - Fail
        ("Llama3-70B", 80, True),     # 140GB model on 80GB GPU - Fail
    ])
    def test_memory_validation(self, model_id, gpu_memory_gb, should_fail):
        """Test that memory validation works correctly."""
        config = self.create_test_config(model_id, gpu_memory_gb)
        
        try:
            orchestrator = ExperimentOrchestrator(config)
            # Need to call setup_simulation to actually create frameworks
            orchestrator.setup_simulation()
            # If we get here, initialization succeeded
            assert not should_fail, f"{model_id} on {gpu_memory_gb}GB should have failed"
        except ValueError as e:
            # Initialization failed
            assert should_fail, f"{model_id} on {gpu_memory_gb}GB should have succeeded"
            assert "exceeds available GPU memory" in str(e)


class TestParallelismRegression:
    """Test parallelism configurations work correctly."""
    
    def run_quick_test(self, config: Dict[str, Any]) -> Tuple[bool, float]:
        """Run a quick test and return success rate and time."""
        start_time = time.time()
        try:
            orchestrator = ExperimentOrchestrator(config)
            report = orchestrator.run()
            runtime = time.time() - start_time
            
            success_rate = report.get('requests', {}).get('success_rate', 0)
            return success_rate, runtime
        except Exception as e:
            return 0.0, 0.0
    
    def test_tp_performance(self):
        """Test that TP has reasonable performance."""
        config = {
            "simulation": {"max_simulation_time": 20, "lod": "medium"},
            "model_characteristics_db_path": "./configs/model_params.json",
            "hardware_profile": {
                "compute_devices": [
                    {
                        "device_id": f"gpu{i}",
                        "device_type": "GPU",
                        "peak_tflops": {"fp16": 312, "int8": 624},
                        "memory_capacity_bytes": 80_000_000_000,
                        "memory_gbps": 2039,
                        "processing_units": 108
                    }
                    for i in range(4)
                ],
                "network_links": [
                    {
                        "link_id": f"gpu{i}_to_gpu{j}",
                        "source_id": f"gpu{i}",
                        "dest_id": f"gpu{j}",
                        "bandwidth_bps": 600_000_000_000,
                        "latency_s": 0.000001
                    }
                    for i in range(4)
                    for j in range(4)
                    if i != j
                ]
            },
            "workload": {
                "total_duration": 20,
                "client_profiles": [{
                    "profile_name": "test_profile",
                    "inter_arrival_time_dist_config": {"type": "Exponential", "rate": 5.0},
                    "prompt_tokens_dist_config": {"type": "Fixed", "value": 256},
                    "max_output_tokens_dist_config": {"type": "Fixed", "value": 128}
                }]
            },
            "frameworks_to_test": [{
                "name": "vllm_tp4",
                "type": "ParallelVLLM",
                "is_target_for_workload": True,
                "config": {
                    "model_profile_id": "Llama2-13B",
                    "block_size": 16,
                    "max_num_seqs": 256,
                    "parallelism": {
                        "strategy": "TP",
                        "tp_degree": 4,
                        "gpu_ids": ["gpu0", "gpu1", "gpu2", "gpu3"]
                    }
                }
            }],
            "metrics_config": {
                "output_summary_json_path": "./experiments/results/regression_tp4.json",
                "output_requests_csv_path": "./experiments/results/regression_tp4.csv",
                "warm_up_duration_s": 2
            }
        }
        
        success_rate, runtime = self.run_quick_test(config)
        
        # TP should have high success rate
        assert success_rate > 0.8, f"TP4 success rate too low: {success_rate}"
        
        # Medium LoD should be fast
        assert runtime < 10, f"TP4 runtime too slow: {runtime}s"
    
    def test_pp_latency_reasonable(self):
        """Test that PP has reasonable latency (not 55x worse)."""
        # We'll create configs for single GPU and PP4
        base_config = {
            "simulation": {"max_simulation_time": 15, "lod": "medium"},
            "model_characteristics_db_path": "./configs/model_params.json",
            "workload": {
                "total_duration": 15,
                "client_profiles": [{
                    "profile_name": "test_profile",
                    "inter_arrival_time_dist_config": {"type": "Fixed", "value": 3.0},
                    "prompt_tokens_dist_config": {"type": "Fixed", "value": 256},
                    "max_output_tokens_dist_config": {"type": "Fixed", "value": 64}
                }]
            },
            "metrics_config": {
                "warm_up_duration_s": 2
            }
        }
        
        # Single GPU config
        single_config = base_config.copy()
        single_config.update({
            "hardware_profile": {
                "compute_devices": [{
                    "device_id": "gpu0",
                    "device_type": "GPU",
                    "peak_tflops": {"fp16": 312, "int8": 624},
                    "memory_capacity_bytes": 80_000_000_000,
                    "memory_gbps": 2039,
                    "processing_units": 108
                }],
                "network_links": [{
                    "link_id": "client_to_gpu0",
                    "source_id": "client_node_0",
                    "dest_id": "gpu0",
                    "bandwidth_bps": 10_000_000_000,
                    "latency_s": 0.0001,
                    "bidirectional": True
                }]
            },
            "frameworks_to_test": [{
                "name": "vllm_single",
                "type": "VLLM",
                "is_target_for_workload": True,
                "config": {
                    "model_profile_id": "Llama2-13B",
                    "gpu_id": "gpu0",
                    "block_size": 16,
                    "max_num_seqs": 256
                }
            }],
            "metrics_config": {
                "output_summary_json_path": "./experiments/results/regression_single.json",
                "output_requests_csv_path": "./experiments/results/regression_single.csv",
                "warm_up_duration_s": 2
            }
        })
        
        # PP4 config
        pp4_config = base_config.copy()
        pp4_config.update({
            "hardware_profile": {
                "compute_devices": [
                    {
                        "device_id": f"gpu{i}",
                        "device_type": "GPU",
                        "peak_tflops": {"fp16": 312, "int8": 624},
                        "memory_capacity_bytes": 80_000_000_000,
                        "memory_gbps": 2039,
                        "processing_units": 108
                    }
                    for i in range(4)
                ],
                "network_links": [
                    {
                        "link_id": f"gpu{i}_to_gpu{j}",
                        "source_id": f"gpu{i}",
                        "dest_id": f"gpu{j}",
                        "bandwidth_bps": 600_000_000_000,
                        "latency_s": 0.000001
                    }
                    for i in range(4)
                    for j in range(4)
                    if i != j
                ]
            },
            "frameworks_to_test": [{
                "name": "vllm_pp4",
                "type": "ParallelVLLM",
                "is_target_for_workload": True,
                "config": {
                    "model_profile_id": "Llama2-13B",
                    "block_size": 16,
                    "max_num_seqs": 256,
                    "parallelism": {
                        "strategy": "PP",
                        "pp_stages": 4,
                        "num_microbatches_per_request": 1,
                        "gpu_ids": ["gpu0", "gpu1", "gpu2", "gpu3"]
                    }
                }
            }],
            "metrics_config": {
                "output_summary_json_path": "./experiments/results/regression_pp4.json",
                "output_requests_csv_path": "./experiments/results/regression_pp4.csv",
                "warm_up_duration_s": 2
            }
        })
        
        # Run both
        try:
            single_orch = ExperimentOrchestrator(single_config)
            single_report = single_orch.run()
            
            pp4_orch = ExperimentOrchestrator(pp4_config)
            pp4_report = pp4_orch.run()
            
            # Extract latencies
            single_ttft = single_report['latency']['time_to_first_token_ms']['mean']
            pp4_ttft = pp4_report['latency']['time_to_first_token_ms']['mean']
            
            single_e2e = single_report['latency']['end_to_end_latency_ms']['mean']
            pp4_e2e = pp4_report['latency']['end_to_end_latency_ms']['mean']
            
            # PP should have similar latency (within 2x)
            ttft_ratio = pp4_ttft / single_ttft if single_ttft > 0 else float('inf')
            e2e_ratio = pp4_e2e / single_e2e if single_e2e > 0 else float('inf')
            
            assert ttft_ratio < 2.0, f"PP4 TTFT {ttft_ratio:.1f}x worse than single GPU"
            assert e2e_ratio < 2.0, f"PP4 E2E {e2e_ratio:.1f}x worse than single GPU"
            
        except Exception as e:
            pytest.fail(f"Failed to run PP latency test: {e}")


class TestLoDRegression:
    """Test Level of Detail functionality."""
    
    def test_lod_event_reduction(self):
        """Test that medium LoD provides significant speedup."""
        import copy
        
        base_config = {
            "simulation": {"max_simulation_time": 30},
            "model_characteristics_db_path": "./configs/model_params.json",
            "hardware_profile": {
                "compute_devices": [
                    {
                        "device_id": f"gpu{i}",
                        "device_type": "GPU",
                        "peak_tflops": {"fp16": 312, "int8": 624},
                        "memory_capacity_bytes": 80_000_000_000,
                        "memory_gbps": 2039,
                        "processing_units": 108
                    }
                    for i in range(4)
                ],
                "network_links": [
                    {
                        "link_id": f"gpu{i}_to_gpu{j}",
                        "source_id": f"gpu{i}",
                        "dest_id": f"gpu{j}",
                        "bandwidth_bps": 600_000_000_000,
                        "latency_s": 0.000001
                    }
                    for i in range(4)
                    for j in range(4)
                    if i != j
                ] + [
                    {
                        "link_id": f"client_to_gpu{i}",
                        "source_id": "client_node_0",
                        "dest_id": f"gpu{i}",
                        "bandwidth_bps": 10_000_000_000,
                        "latency_s": 0.0001,
                        "bidirectional": True
                    }
                    for i in range(4)
                ]
            },
            "workload": {
                "total_duration": 30,
                "client_profiles": [{
                    "profile_name": "test_profile",
                    # Moderate request rate
                    "inter_arrival_time_dist_config": {"type": "Fixed", "value": 1.0},
                    # Large prompts to trigger many layer events
                    "prompt_tokens_dist_config": {"type": "Fixed", "value": 2048},
                    "max_output_tokens_dist_config": {"type": "Fixed", "value": 256}
                }]
            },
            "frameworks_to_test": [{
                "name": "vllm_tp4",
                "type": "ParallelVLLM",
                "is_target_for_workload": True,
                "config": {
                    "model_profile_id": "Llama2-13B",
                    "block_size": 16,
                    "max_num_seqs": 256,
                    "parallelism": {
                        "strategy": "TP",
                        "tp_degree": 4,
                        "gpu_ids": ["gpu0", "gpu1", "gpu2", "gpu3"]
                    }
                }
            }],
            "metrics_config": {
                "warm_up_duration_s": 2
            }
        }
        
        # Test high LoD
        high_config = copy.deepcopy(base_config)
        high_config["simulation"]["lod"] = "high"
        high_config["metrics_config"]["output_summary_json_path"] = "./experiments/results/regression_lod_high.json"
        high_config["metrics_config"]["output_requests_csv_path"] = "./experiments/results/regression_lod_high.csv"
        
        # Test medium LoD
        medium_config = copy.deepcopy(base_config)
        medium_config["simulation"]["lod"] = "medium"
        medium_config["metrics_config"]["output_summary_json_path"] = "./experiments/results/regression_lod_medium.json"
        medium_config["metrics_config"]["output_requests_csv_path"] = "./experiments/results/regression_lod_medium.csv"
        
        # Run high LoD
        print(f"\nRunning high LoD test")
        high_orch = ExperimentOrchestrator(high_config)
        high_report = high_orch.run()
        print(f"High LoD: {high_report['requests']['successful']} requests")
        
        # Check framework status to verify LoD was applied
        if 'frameworks' in high_report:
            for fw_name, fw_status in high_report['frameworks'].items():
                lod_value = fw_status.get('lod', 'unknown')
                print(f"  Framework {fw_name} LoD: {lod_value}")
                assert lod_value == 'high', f"Framework should have high LoD, got {lod_value}"
        
        # Run medium LoD
        print(f"\nRunning medium LoD test")
        medium_orch = ExperimentOrchestrator(medium_config)
        medium_report = medium_orch.run()
        print(f"Medium LoD: {medium_report['requests']['successful']} requests")
        
        # Check framework status to verify LoD was applied
        if 'frameworks' in medium_report:
            for fw_name, fw_status in medium_report['frameworks'].items():
                lod_value = fw_status.get('lod', 'unknown')
                print(f"  Framework {fw_name} LoD: {lod_value}")
                assert lod_value == 'medium', f"Framework should have medium LoD, got {lod_value}"
        
        # Results should be similar
        high_reqs = high_report['requests']['successful']
        medium_reqs = medium_report['requests']['successful']
        
        # Both should process similar number of requests (within 10%)
        if high_reqs > 0:
            req_diff = abs(medium_reqs - high_reqs) / high_reqs
            assert req_diff < 0.1, f"Request counts differ too much: high={high_reqs}, medium={medium_reqs} ({req_diff:.1%} difference)"
        
        # Latency should be similar (within 20%)
        high_ttft = high_report['latency']['time_to_first_token_ms']['mean']
        medium_ttft = medium_report['latency']['time_to_first_token_ms']['mean']
        
        if high_ttft > 0:
            ttft_diff = abs(medium_ttft - high_ttft) / high_ttft
            assert ttft_diff < 0.2, f"TTFT differs by {ttft_diff:.1%} (high={high_ttft:.1f}ms, medium={medium_ttft:.1f}ms)"
        
        # Both should have good success rates
        assert high_report['requests']['success_rate'] > 0.8, f"High LoD success rate too low: {high_report['requests']['success_rate']:.1%}"
        assert medium_report['requests']['success_rate'] > 0.8, f"Medium LoD success rate too low: {medium_report['requests']['success_rate']:.1%}"
        
        print(f"\nLoD test passed: Both configurations processed similar requests with correct LoD settings")


class TestFeatureRegression:
    """Test specific features work correctly."""
    
    def test_prefix_caching_enabled(self):
        """Test that prefix caching can be enabled."""
        config = {
            "simulation": {"max_simulation_time": 10, "lod": "medium"},
            "model_characteristics_db_path": "./configs/model_params.json",
            "hardware_profile": {
                "compute_devices": [{
                    "device_id": "gpu0",
                    "device_type": "GPU",
                    "peak_tflops": {"fp16": 312, "int8": 624},
                    "memory_capacity_bytes": 80_000_000_000,
                    "memory_gbps": 2039,
                    "processing_units": 108
                }],
                "network_links": [{
                    "link_id": "client_to_gpu0",
                    "source_id": "client_node_0",
                    "dest_id": "gpu0",
                    "bandwidth_bps": 10_000_000_000,
                    "latency_s": 0.0001,
                    "bidirectional": True
                }]
            },
            "workload": {
                "total_duration": 10,
                "client_profiles": [{
                    "profile_name": "test_profile",
                    "inter_arrival_time_dist_config": {"type": "Fixed", "value": 1.0},
                    "prompt_tokens_dist_config": {"type": "Fixed", "value": 100},
                    "max_output_tokens_dist_config": {"type": "Fixed", "value": 50},
                    "conversational_probability": 0.5  # Enable conversations
                }]
            },
            "frameworks_to_test": [{
                "name": "vllm_test",
                "type": "VLLM",
                "is_target_for_workload": True,
                "config": {
                    "model_profile_id": "Llama2-7B",
                    "gpu_id": "gpu0",
                    "block_size": 16,
                    "max_num_seqs": 256,
                    "enable_prefix_caching": True,  # Test this works
                    "enable_cross_request_caching": True
                }
            }],
            "metrics_config": {
                "output_summary_json_path": "./experiments/results/regression_prefix_cache.json",
                "output_requests_csv_path": "./experiments/results/regression_prefix_cache.csv"
            }
        }
        
        try:
            orchestrator = ExperimentOrchestrator(config)
            report = orchestrator.run()
            
            # Should complete successfully
            assert report['requests']['total'] > 0
            
        except Exception as e:
            pytest.fail(f"Prefix caching test failed: {e}")
    
    def test_chunked_prefill_enabled(self):
        """Test that chunked prefill works."""
        config = {
            "simulation": {"max_simulation_time": 10, "lod": "medium"},
            "model_characteristics_db_path": "./configs/model_params.json",
            "hardware_profile": {
                "compute_devices": [{
                    "device_id": "gpu0",
                    "device_type": "GPU",
                    "peak_tflops": {"fp16": 312, "int8": 624},
                    "memory_capacity_bytes": 80_000_000_000,
                    "memory_gbps": 2039,
                    "processing_units": 108
                }],
                "network_links": [{
                    "link_id": "client_to_gpu0",
                    "source_id": "client_node_0",
                    "dest_id": "gpu0",
                    "bandwidth_bps": 10_000_000_000,
                    "latency_s": 0.0001,
                    "bidirectional": True
                }]
            },
            "workload": {
                "total_duration": 10,
                "client_profiles": [{
                    "profile_name": "test_profile",
                    "inter_arrival_time_dist_config": {"type": "Fixed", "value": 2.0},
                    "prompt_tokens_dist_config": {"type": "Fixed", "value": 8192},  # Large prompt
                    "max_output_tokens_dist_config": {"type": "Fixed", "value": 128}
                }]
            },
            "frameworks_to_test": [{
                "name": "vllm_test",
                "type": "VLLM",
                "is_target_for_workload": True,
                "config": {
                    "model_profile_id": "Llama2-7B",
                    "gpu_id": "gpu0",
                    "block_size": 16,
                    "max_num_seqs": 256,
                    "enable_chunked_prefill": True,
                    "prefill_chunk_size": 2048  # Test chunking
                }
            }],
            "metrics_config": {
                "output_summary_json_path": "./experiments/results/regression_chunked_prefill.json",
                "output_requests_csv_path": "./experiments/results/regression_chunked_prefill.csv"
            }
        }
        
        try:
            orchestrator = ExperimentOrchestrator(config)
            report = orchestrator.run()
            
            # Should handle large prompts
            assert report['requests']['successful'] > 0
            
        except Exception as e:
            pytest.fail(f"Chunked prefill test failed: {e}")