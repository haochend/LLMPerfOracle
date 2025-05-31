"""Improved integration tests for multi-GPU parallel simulations.

These tests create realistic scenarios where parallelism provides actual benefits.
"""

import json
import numpy as np
from pathlib import Path

import pytest

from llmperforacle.orchestration import ExperimentOrchestrator


class TestRealisticParallelScenarios:
    """Test parallelism with realistic workloads that demonstrate benefits."""
    
    @pytest.fixture
    def heavy_workload_config(self):
        """Configuration with heavy workload that stresses the system."""
        return {
            "simulation": {
                "max_simulation_time": 60,  # Longer simulation
                "random_seed": 42
            },
            "model_characteristics_db_path": "configs/model_params.json",
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
                    for i in range(8)
                ],
                "network_links": self._generate_realistic_network(8)
            },
            "workload": {
                "total_duration": 60,
                "bytes_per_token_estimate_for_network": 2,
                "client_profiles": [
                    {
                        "profile_name": "heavy_compute",
                        "weight": 0.3,
                        "inter_arrival_time_dist_config": {
                            "type": "Exponential",
                            "rate": 5.0  # 5 requests per second
                        },
                        "prompt_tokens_dist_config": {
                            "type": "LogNormal",
                            "mean": 1500,  # Large prompts
                            "sigma": 500,
                            "is_int": True
                        },
                        "max_output_tokens_dist_config": {
                            "type": "Uniform",
                            "low": 50,
                            "high": 200,
                            "is_int": True
                        },
                        "conversational_probability": 0.0,
                        "streaming_response_probability": 0.8
                    },
                    {
                        "profile_name": "memory_intensive",
                        "weight": 0.4,
                        "inter_arrival_time_dist_config": {
                            "type": "Exponential",
                            "rate": 10.0  # 10 requests per second
                        },
                        "prompt_tokens_dist_config": {
                            "type": "Uniform",
                            "low": 200,
                            "high": 500,
                            "is_int": True
                        },
                        "max_output_tokens_dist_config": {
                            "type": "LogNormal",
                            "mean": 1000,  # Long outputs
                            "sigma": 300,
                            "is_int": True
                        },
                        "conversational_probability": 0.5,
                        "streaming_response_probability": 0.9
                    },
                    {
                        "profile_name": "bursty_traffic",
                        "weight": 0.3,
                        "inter_arrival_time_dist_config": {
                            "type": "LogNormal",
                            "mean": 0.1,  # Bursty pattern
                            "sigma": 0.5
                        },
                        "prompt_tokens_dist_config": {
                            "type": "Uniform",
                            "low": 100,
                            "high": 300,
                            "is_int": True
                        },
                        "max_output_tokens_dist_config": {
                            "type": "Constant",
                            "value": 128,
                            "is_int": True
                        },
                        "conversational_probability": 0.2,
                        "streaming_response_probability": 0.5
                    }
                ]
            },
            "metrics_config": {
                "percentiles_to_calculate": [0.5, 0.9, 0.95, 0.99],
                "warm_up_duration_s": 10
            }
        }
    
    def _generate_realistic_network(self, num_gpus):
        """Generate realistic network topology (not full mesh)."""
        links = []
        
        # Ring topology for GPUs (more realistic than full mesh)
        for i in range(num_gpus):
            next_gpu = (i + 1) % num_gpus
            links.append({
                "link_id": f"gpu{i}_to_gpu{next_gpu}",
                "source_id": f"gpu{i}",
                "dest_id": f"gpu{next_gpu}",
                "bandwidth_bps": 600_000_000_000,  # 600 Gbps NVLink
                "latency_s": 0.0000005,
                "bidirectional": True
            })
        
        # Add some cross-connections for better connectivity
        # This ensures any GPU can reach any other GPU with at most 2 hops
        for i in range(0, num_gpus, 2):
            if i + 2 < num_gpus:
                links.append({
                    "link_id": f"gpu{i}_to_gpu{i+2}",
                    "source_id": f"gpu{i}",
                    "dest_id": f"gpu{i+2}",
                    "bandwidth_bps": 300_000_000_000,  # 300 Gbps (slower)
                    "latency_s": 0.000001,  # 1 µs
                    "bidirectional": True
                })
        
        # Add diagonal connections for 4x4 and larger topologies
        if num_gpus >= 4:
            # Connect opposite corners for better connectivity
            links.append({
                "link_id": f"gpu0_to_gpu{num_gpus-1}",
                "source_id": "gpu0",
                "dest_id": f"gpu{num_gpus-1}",
                "bandwidth_bps": 300_000_000_000,
                "latency_s": 0.000001,
                "bidirectional": True
            })
        
        # Client to framework link
        links.append({
            "link_id": "client_to_server",
            "source_id": "client_node_0",
            "dest_id": "framework_entry_0",
            "bandwidth_bps": 10_000_000_000,
            "latency_s": 0.0001,
            "bidirectional": True
        })
        
        return links
    
    def test_tp_for_compute_bound_workload(self, heavy_workload_config, tmp_path):
        """Test TP benefits for compute-bound (large prompt) workloads."""
        config = heavy_workload_config.copy()
        
        # Focus on compute-heavy profile
        config["workload"]["client_profiles"] = [{
            "profile_name": "compute_heavy",
            "weight": 1.0,
            "inter_arrival_time_dist_config": {
                "type": "Exponential",
                "rate": 3.0  # 3 req/s to ensure system is loaded
            },
            "prompt_tokens_dist_config": {
                "type": "Constant",
                "value": 2000,  # Very large prompts
                "is_int": True
            },
            "max_output_tokens_dist_config": {
                "type": "Constant",
                "value": 50,  # Short outputs
                "is_int": True
            },
            "conversational_probability": 0.0,
            "streaming_response_probability": 0.0
        }]
        
        results = {}
        
        # Test different TP degrees
        for tp in [1, 2, 4]:
            test_config = config.copy()
            test_config["frameworks_to_test"] = [{
                "name": f"vllm_tp{tp}",
                "type": "VLLM",
                "is_target_for_workload": True,
                "config": {
                    "model_profile_id": "Llama2-13B",  # Larger model
                    "gpu_id": "gpu0",
                    "block_size": 16,
                    "max_num_seqs": 16,  # Keep constant
                    "max_num_batched_tokens": 8192,
                    "scheduler_iteration_delay_s": 0.0001
                }
            }]
            
            if tp > 1:
                test_config["frameworks_to_test"][0]["config"]["parallelism"] = {
                    "strategy": "TP",
                    "tp_degree": tp,
                    "gpu_ids": [f"gpu{i}" for i in range(tp)]
                }
            
            test_config["metrics_config"]["output_summary_json_path"] = str(tmp_path / f"tp{tp}_compute.json")
            test_config["metrics_config"]["output_requests_csv_path"] = str(tmp_path / f"tp{tp}_compute.csv")
            
            orchestrator = ExperimentOrchestrator(test_config)
            orchestrator.run()
            
            with open(tmp_path / f"tp{tp}_compute.json") as f:
                results[tp] = json.load(f)
        
        # Verify TP scaling for compute-bound workload
        # TTFT should improve significantly with TP
        ttft_improvement_2x = results[1]["latency"]["time_to_first_token_ms"]["p50"] / \
                              results[2]["latency"]["time_to_first_token_ms"]["p50"]
        ttft_improvement_4x = results[1]["latency"]["time_to_first_token_ms"]["p50"] / \
                              results[4]["latency"]["time_to_first_token_ms"]["p50"]
        
        # Expect at least 1.5x improvement with TP=2 (accounting for overhead)
        assert ttft_improvement_2x > 1.5, f"TP=2 TTFT improvement only {ttft_improvement_2x:.2f}x"
        # Expect at least 2.5x improvement with TP=4
        assert ttft_improvement_4x > 2.5, f"TP=4 TTFT improvement only {ttft_improvement_4x:.2f}x"
        
        # Success rate should remain high for parallel configs
        # Single GPU might struggle with 2000 token prompts at 3 req/s
        assert results[1]["requests"]["success_rate"] > 0.7  # Lower threshold for single GPU
        assert results[2]["requests"]["success_rate"] > 0.85  # Better with TP=2
        assert results[4]["requests"]["success_rate"] > 0.9   # Should handle well with TP=4
    
    def test_pp_for_memory_constrained_workload(self, heavy_workload_config, tmp_path):
        """Test PP benefits for memory-constrained scenarios."""
        config = heavy_workload_config.copy()
        
        # Configure for memory-intensive workload
        config["workload"]["client_profiles"] = [{
            "profile_name": "memory_heavy",
            "weight": 1.0,
            "inter_arrival_time_dist_config": {
                "type": "Exponential",
                "rate": 8.0  # Higher rate
            },
            "prompt_tokens_dist_config": {
                "type": "Uniform",
                "low": 500,
                "high": 1000,
                "is_int": True
            },
            "max_output_tokens_dist_config": {
                "type": "Constant",
                "value": 1500,  # Very long outputs
                "is_int": True
            },
            "conversational_probability": 0.0,
            "streaming_response_probability": 1.0
        }]
        
        # Reduce memory per GPU to force memory constraints
        for device in config["hardware_profile"]["compute_devices"]:
            device["memory_capacity_bytes"] = 40_000_000_000  # 40GB instead of 80GB
        
        results = {}
        
        # Test single GPU vs PP
        for pp in [1, 2, 4]:
            test_config = config.copy()
            
            if pp == 1:
                # Single GPU - should struggle with memory
                test_config["frameworks_to_test"] = [{
                    "name": "vllm_single",
                    "type": "VLLM",
                    "is_target_for_workload": True,
                    "config": {
                        "model_profile_id": "Llama2-13B",
                        "gpu_id": "gpu0",
                        "block_size": 16,
                        "max_num_seqs": 8,  # Limited by memory
                        "max_num_batched_tokens": 4096,
                        "scheduler_iteration_delay_s": 0.0001
                    }
                }]
            else:
                # Pipeline parallel - can handle more sequences
                test_config["frameworks_to_test"] = [{
                    "name": f"parallel_vllm_pp{pp}",
                    "type": "ParallelVLLM",
                    "is_target_for_workload": True,
                    "config": {
                        "model_profile_id": "Llama2-13B",
                        "gpu_id": "gpu0",
                        "block_size": 16,
                        "max_num_seqs": 8 * pp,  # More sequences with PP
                        "parallelism": {
                            "strategy": "PP",
                            "pp_stages": pp,
                            "num_microbatches_per_request": pp * 2,
                            "gpu_ids": [f"gpu{i}" for i in range(pp)]
                        }
                    }
                }]
            
            test_config["metrics_config"]["output_summary_json_path"] = str(tmp_path / f"pp{pp}_memory.json")
            test_config["metrics_config"]["output_requests_csv_path"] = str(tmp_path / f"pp{pp}_memory.csv")
            
            orchestrator = ExperimentOrchestrator(test_config)
            orchestrator.run()
            
            with open(tmp_path / f"pp{pp}_memory.json") as f:
                results[pp] = json.load(f)
        
        # PP should handle more requests successfully
        assert results[2]["requests"]["total"] > results[1]["requests"]["total"] * 1.5
        assert results[4]["requests"]["total"] > results[2]["requests"]["total"] * 1.3
        
        # PP should have better success rate under memory pressure
        assert results[2]["requests"]["success_rate"] > results[1]["requests"]["success_rate"]
        assert results[4]["requests"]["success_rate"] > results[1]["requests"]["success_rate"]
    
    def test_dp_for_high_throughput(self, heavy_workload_config, tmp_path):
        """Test DP for handling high request rates."""
        config = heavy_workload_config.copy()
        
        # Very high request rate
        config["workload"]["client_profiles"] = [{
            "profile_name": "high_throughput",
            "weight": 1.0,
            "inter_arrival_time_dist_config": {
                "type": "Exponential",
                "rate": 50.0  # 50 requests per second!
            },
            "prompt_tokens_dist_config": {
                "type": "Uniform",
                "low": 100,
                "high": 300,
                "is_int": True
            },
            "max_output_tokens_dist_config": {
                "type": "Uniform",
                "low": 50,
                "high": 150,
                "is_int": True
            },
            "conversational_probability": 0.3,
            "streaming_response_probability": 0.5
        }]
        
        results = {}
        
        # Test different numbers of replicas
        for num_replicas in [1, 2, 4]:
            test_config = config.copy()
            test_config["frameworks_to_test"] = []
            
            for i in range(num_replicas):
                test_config["frameworks_to_test"].append({
                    "name": f"vllm_dp{i}",
                    "type": "VLLM",
                    "is_target_for_workload": True,
                    "config": {
                        "model_profile_id": "Llama2-7B",
                        "gpu_id": f"gpu{i}",
                        "block_size": 16,
                        "max_num_seqs": 64,
                        "max_num_batched_tokens": 4096,
                        "scheduler_iteration_delay_s": 0.0001
                    }
                })
            
            # Use least_loaded for better distribution
            test_config["workload"]["load_balancing_strategy"] = "least_loaded"
            
            test_config["metrics_config"]["output_summary_json_path"] = str(tmp_path / f"dp{num_replicas}_throughput.json")
            test_config["metrics_config"]["output_requests_csv_path"] = str(tmp_path / f"dp{num_replicas}_throughput.csv")
            
            orchestrator = ExperimentOrchestrator(test_config)
            orchestrator.run()
            
            with open(tmp_path / f"dp{num_replicas}_throughput.json") as f:
                results[num_replicas] = json.load(f)
        
        # Verify throughput scaling
        throughput_1 = results[1]["throughput"]["requests_per_second"]
        throughput_2 = results[2]["throughput"]["requests_per_second"]
        throughput_4 = results[4]["throughput"]["requests_per_second"]
        
        # Should see near-linear scaling for DP
        assert throughput_2 > throughput_1 * 1.8, f"DP=2 only {throughput_2/throughput_1:.2f}x throughput"
        assert throughput_4 > throughput_1 * 3.5, f"DP=4 only {throughput_4/throughput_1:.2f}x throughput"
        
        # Latency should remain reasonable
        assert results[4]["latency"]["time_to_first_token_ms"]["p99"] < 1000  # Under 1 second
    
    def test_combined_parallelism_for_large_model(self, heavy_workload_config, tmp_path):
        """Test combined TP+PP for large models that don't fit on single GPU."""
        config = heavy_workload_config.copy()
        
        # Mixed workload
        config["workload"]["client_profiles"] = [
            {
                "profile_name": "large_prompts",
                "weight": 0.5,
                "inter_arrival_time_dist_config": {"type": "Exponential", "rate": 2.0},
                "prompt_tokens_dist_config": {"type": "Uniform", "low": 1000, "high": 2000, "is_int": True},
                "max_output_tokens_dist_config": {"type": "Constant", "value": 100, "is_int": True},
                "conversational_probability": 0.0,
                "streaming_response_probability": 0.0
            },
            {
                "profile_name": "long_generation",
                "weight": 0.5,
                "inter_arrival_time_dist_config": {"type": "Exponential", "rate": 3.0},
                "prompt_tokens_dist_config": {"type": "Constant", "value": 200, "is_int": True},
                "max_output_tokens_dist_config": {"type": "Uniform", "low": 500, "high": 1500, "is_int": True},
                "conversational_probability": 0.0,
                "streaming_response_probability": 1.0
            }
        ]
        
        results = {}
        
        # Test configurations
        configs_to_test = [
            ("single", None, None),  # Baseline
            ("tp4", 4, None),       # TP only
            ("pp4", None, 4),       # PP only
            ("tp2_pp2", 2, 2),      # Combined
        ]
        
        for name, tp, pp in configs_to_test:
            test_config = config.copy()
            
            if name == "single":
                # Single GPU baseline - will struggle
                test_config["frameworks_to_test"] = [{
                    "name": "vllm_single",
                    "type": "VLLM",
                    "is_target_for_workload": True,
                    "config": {
                        "model_profile_id": "GPT-3-175B",  # Very large model
                        "gpu_id": "gpu0",
                        "block_size": 16,
                        "max_num_seqs": 4,  # Very limited
                        "max_num_batched_tokens": 512,
                        "scheduler_iteration_delay_s": 0.0001
                    }
                }]
            elif pp is None:
                # TP only
                test_config["frameworks_to_test"] = [{
                    "name": f"vllm_{name}",
                    "type": "VLLM",
                    "is_target_for_workload": True,
                    "config": {
                        "model_profile_id": "GPT-3-175B",
                        "gpu_id": "gpu0",
                        "block_size": 16,
                        "max_num_seqs": 4 * tp,
                        "max_num_batched_tokens": 512 * tp,
                        "scheduler_iteration_delay_s": 0.0001,
                        "parallelism": {
                            "strategy": "TP",
                            "tp_degree": tp,
                            "gpu_ids": [f"gpu{i}" for i in range(tp)]
                        }
                    }
                }]
            elif tp is None:
                # PP only
                test_config["frameworks_to_test"] = [{
                    "name": f"parallel_vllm_{name}",
                    "type": "ParallelVLLM",
                    "is_target_for_workload": True,
                    "config": {
                        "model_profile_id": "GPT-3-175B",
                        "gpu_id": "gpu0",
                        "block_size": 16,
                        "max_num_seqs": 4 * pp,
                        "parallelism": {
                            "strategy": "PP",
                            "pp_stages": pp,
                            "num_microbatches_per_request": pp * 2,
                            "gpu_ids": [f"gpu{i}" for i in range(pp)]
                        }
                    }
                }]
            else:
                # Combined TP+PP
                test_config["frameworks_to_test"] = [{
                    "name": f"parallel_vllm_{name}",
                    "type": "ParallelVLLM",
                    "is_target_for_workload": True,
                    "config": {
                        "model_profile_id": "GPT-3-175B",
                        "gpu_id": "gpu0",
                        "block_size": 16,
                        "max_num_seqs": 16,  # 4 * (tp * pp)
                        "parallelism": {
                            "strategy": "TP_PP",
                            "tp_degree": tp,
                            "pp_stages": pp,
                            "num_microbatches_per_request": 4,
                            "gpu_ids": ["gpu0", "gpu1", "gpu2", "gpu3"]
                        }
                    }
                }]
            
            # Reduce simulation time for large model
            test_config["simulation"]["max_simulation_time"] = 30
            test_config["workload"]["total_duration"] = 30
            
            test_config["metrics_config"]["output_summary_json_path"] = str(tmp_path / f"{name}_combined.json")
            test_config["metrics_config"]["output_requests_csv_path"] = str(tmp_path / f"{name}_combined.csv")
            
            orchestrator = ExperimentOrchestrator(test_config)
            orchestrator.run()
            
            with open(tmp_path / f"{name}_combined.json") as f:
                results[name] = json.load(f)
        
        # Verify parallelism helps with large model
        # Single GPU should struggle
        assert results["single"]["requests"]["success_rate"] < 0.5
        
        # All parallel configs should do better
        for config in ["tp4", "pp4", "tp2_pp2"]:
            assert results[config]["requests"]["success_rate"] > 0.8
            assert results[config]["requests"]["total"] > results["single"]["requests"]["total"]
        
        # Combined should have good balance
        # TP helps with compute, PP helps with memory
        assert results["tp2_pp2"]["latency"]["time_to_first_token_ms"]["p50"] < \
               results["pp4"]["latency"]["time_to_first_token_ms"]["p50"]
        assert results["tp2_pp2"]["requests"]["total"] >= results["tp4"]["requests"]["total"] * 0.9


class TestParallelismMetrics:
    """Test that parallelism metrics are properly tracked."""
    
    def test_gpu_utilization_tracking(self, tmp_path):
        """Verify GPU utilization is tracked correctly across parallel configs."""
        config = {
            "simulation": {"max_simulation_time": 30, "random_seed": 42},
            "model_characteristics_db_path": "configs/model_params.json",
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
                        "link_id": f"gpu{i}_to_gpu{(i+1)%4}",
                        "source_id": f"gpu{i}",
                        "dest_id": f"gpu{(i+1)%4}",
                        "bandwidth_bps": 600_000_000_000,
                        "latency_s": 0.0000005,
                        "bidirectional": True
                    }
                    for i in range(4)
                ] + [{
                    "link_id": "client_to_server",
                    "source_id": "client_node_0",
                    "dest_id": "framework_entry_0",
                    "bandwidth_bps": 10_000_000_000,
                    "latency_s": 0.0001,
                    "bidirectional": True
                }]
            },
            "workload": {
                "total_duration": 30,
                "bytes_per_token_estimate_for_network": 2,
                "client_profiles": [{
                    "profile_name": "steady",
                    "weight": 1.0,
                    "inter_arrival_time_dist_config": {"type": "Constant", "value": 0.2},
                    "prompt_tokens_dist_config": {"type": "Constant", "value": 500, "is_int": True},
                    "max_output_tokens_dist_config": {"type": "Constant", "value": 200, "is_int": True},
                    "conversational_probability": 0.0,
                    "streaming_response_probability": 0.0
                }]
            },
            "metrics_config": {
                "percentiles_to_calculate": [0.5, 0.9, 0.99],
                "warm_up_duration_s": 5
            }
        }
        
        # Test TP=4 configuration
        config["frameworks_to_test"] = [{
            "name": "vllm_tp4",
            "type": "VLLM",
            "is_target_for_workload": True,
            "config": {
                "model_profile_id": "Llama2-7B",
                "gpu_id": "gpu0",
                "block_size": 16,
                "max_num_seqs": 32,
                "max_num_batched_tokens": 4096,
                "scheduler_iteration_delay_s": 0.0001,
                "parallelism": {
                    "strategy": "TP",
                    "tp_degree": 4,
                    "gpu_ids": ["gpu0", "gpu1", "gpu2", "gpu3"]
                }
            }
        }]
        
        config["metrics_config"]["output_summary_json_path"] = str(tmp_path / "gpu_util.json")
        config["metrics_config"]["output_requests_csv_path"] = str(tmp_path / "gpu_util.csv")
        
        orchestrator = ExperimentOrchestrator(config)
        orchestrator.run()
        
        with open(tmp_path / "gpu_util.json") as f:
            results = json.load(f)
        
        # Check GPU utilization metrics
        gpu_utils = results.get("gpu_utilization", {})
        
        # All 4 GPUs should be utilized
        assert len(gpu_utils) >= 4, "Not all GPUs tracked"
        
        # Utilization should be balanced across GPUs in TP
        utils = [gpu_utils.get(f"gpu{i}", 0) for i in range(4)]
        avg_util = np.mean(utils)
        
        # All GPUs should have similar utilization (within 20% of average)
        for i, util in enumerate(utils):
            assert abs(util - avg_util) / avg_util < 0.2, f"GPU{i} utilization {util} too far from average {avg_util}"
        
        # Overall utilization should be reasonable
        assert avg_util > 0.3, f"Average GPU utilization too low: {avg_util}"
        assert avg_util < 0.95, f"Average GPU utilization suspiciously high: {avg_util}"