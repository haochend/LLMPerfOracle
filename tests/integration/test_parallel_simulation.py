"""Integration tests for multi-GPU parallel simulations."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from llmperforacle.orchestration import ExperimentOrchestrator


class TestParallelSimulations:
    """Run actual simulations with different parallelism configurations."""
    
    @pytest.fixture
    def base_config(self):
        """Base configuration for parallel simulations."""
        return {
            "simulation": {
                "max_simulation_time": 10,
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
                    for i in range(8)  # 8 GPUs for various configs
                ],
                "network_links": self._generate_mesh_network(8)
            },
            "workload": {
                "total_duration": 10,
                "bytes_per_token_estimate_for_network": 2,
                "client_profiles": [{
                    "profile_name": "test_profile",
                    "weight": 1.0,
                    "inter_arrival_time_dist_config": {
                        "type": "Exponential",
                        "rate": 1.0  # 1 request per second
                    },
                    "prompt_tokens_dist_config": {
                        "type": "Uniform",
                        "low": 100,
                        "high": 200,
                        "is_int": True
                    },
                    "max_output_tokens_dist_config": {
                        "type": "Constant",
                        "value": 100,
                        "is_int": True
                    },
                    "conversational_probability": 0.0,
                    "streaming_response_probability": 0.0
                }]
            },
            "metrics_config": {
                "percentiles_to_calculate": [0.5, 0.9, 0.99],
                "warm_up_duration_s": 2
            }
        }
    
    def _generate_mesh_network(self, num_gpus):
        """Generate all-to-all mesh network for GPUs."""
        links = []
        # Inter-GPU links (NVLink-like)
        for i in range(num_gpus):
            for j in range(i + 1, num_gpus):
                links.append({
                    "link_id": f"gpu{i}_to_gpu{j}",
                    "source_id": f"gpu{i}",
                    "dest_id": f"gpu{j}",
                    "bandwidth_bps": 600_000_000_000,  # 600 Gbps
                    "latency_s": 0.0000005,  # 0.5 µs
                    "bidirectional": True
                })
        
        # Client to framework link
        links.append({
            "link_id": "client_to_server",
            "source_id": "client_node_0",
            "dest_id": "framework_entry_0",
            "bandwidth_bps": 10_000_000_000,  # 10 Gbps
            "latency_s": 0.0001,
            "bidirectional": True
        })
        
        return links
    
    def test_single_gpu_baseline(self, base_config, tmp_path):
        """Test single GPU performance as baseline."""
        config = base_config.copy()
        config["frameworks_to_test"] = [{
            "name": "vllm_single",
            "type": "VLLM",
            "is_target_for_workload": True,
            "config": {
                "model_profile_id": "Llama2-7B",
                "gpu_id": "gpu0",
                "block_size": 16,
                "max_num_seqs": 32,
                "max_num_batched_tokens": 2048,
                "scheduler_iteration_delay_s": 0.0001
            }
        }]
        
        # Set output paths
        config["metrics_config"]["output_summary_json_path"] = str(tmp_path / "single_gpu_summary.json")
        config["metrics_config"]["output_requests_csv_path"] = str(tmp_path / "single_gpu_requests.csv")
        
        # Run simulation
        orchestrator = ExperimentOrchestrator(config)
        orchestrator.run()
        
        # Load and verify results
        with open(tmp_path / "single_gpu_summary.json") as f:
            results = json.load(f)
        
        assert results["requests"]["success_rate"] > 0.9
        assert results["throughput"]["output_tokens_per_second"] > 0
        
        # Store baseline metrics for comparison
        return results
    
    def test_tensor_parallelism_scaling(self, base_config, tmp_path):
        """Test performance scaling with tensor parallelism."""
        tp_degrees = [1, 2, 4]
        results = {}
        
        for tp in tp_degrees:
            config = base_config.copy()
            config["frameworks_to_test"] = [{
                "name": f"vllm_tp{tp}",
                "type": "VLLM",
                "is_target_for_workload": True,
                "config": {
                    "model_profile_id": "Llama2-7B",
                    "gpu_id": "gpu0",
                    "block_size": 16,
                    "max_num_seqs": 32 * tp,  # Scale with TP
                    "max_num_batched_tokens": 2048 * tp,
                    "scheduler_iteration_delay_s": 0.0001,
                    "parallelism": {
                        "strategy": "TP",
                        "tp_degree": tp,
                        "gpu_ids": [f"gpu{i}" for i in range(tp)]
                    }
                }
            }]
            
            config["metrics_config"]["output_summary_json_path"] = str(tmp_path / f"tp{tp}_summary.json")
            config["metrics_config"]["output_requests_csv_path"] = str(tmp_path / f"tp{tp}_requests.csv")
            
            orchestrator = ExperimentOrchestrator(config)
            orchestrator.run()
            
            with open(tmp_path / f"tp{tp}_summary.json") as f:
                results[tp] = json.load(f)
        
        # Verify scaling behavior
        # TTFT should decrease with more GPUs (parallel computation)
        assert results[2]["latency"]["time_to_first_token_ms"]["p50"] < \
               results[1]["latency"]["time_to_first_token_ms"]["p50"]
        
        # Throughput should increase
        assert results[4]["throughput"]["output_tokens_per_second"] > \
               results[2]["throughput"]["output_tokens_per_second"]
    
    def test_pipeline_parallelism_stages(self, base_config, tmp_path):
        """Test pipeline parallelism with different stage counts."""
        pp_stages = [1, 2, 4]
        results = {}
        
        for pp in pp_stages:
            config = base_config.copy()
            config["frameworks_to_test"] = [{
                "name": f"parallel_vllm_pp{pp}",
                "type": "ParallelVLLM",
                "is_target_for_workload": True,
                "config": {
                    "model_profile_id": "Llama2-7B",
                    "gpu_id": "gpu0",
                    "block_size": 16,
                    "max_num_seqs": 64,
                    "parallelism": {
                        "strategy": "PP",
                        "pp_stages": pp,
                        "num_microbatches_per_request": pp * 2,
                        "gpu_ids": [f"gpu{i}" for i in range(pp)]
                    }
                }
            }]
            
            config["metrics_config"]["output_summary_json_path"] = str(tmp_path / f"pp{pp}_summary.json")
            config["metrics_config"]["output_requests_csv_path"] = str(tmp_path / f"pp{pp}_requests.csv")
            
            orchestrator = ExperimentOrchestrator(config)
            orchestrator.run()
            
            with open(tmp_path / f"pp{pp}_summary.json") as f:
                results[pp] = json.load(f)
        
        # PP should allow handling more concurrent requests
        assert results[4]["requests"]["total"] >= results[1]["requests"]["total"]
    
    def test_data_parallelism_load_balancing(self, base_config, tmp_path):
        """Test data parallelism with different load balancing strategies."""
        strategies = ["round_robin", "least_loaded", "random"]
        results = {}
        
        for strategy in strategies:
            config = base_config.copy()
            config["workload"]["load_balancing_strategy"] = strategy
            
            # Create 4 identical instances for DP
            config["frameworks_to_test"] = []
            for i in range(4):
                config["frameworks_to_test"].append({
                    "name": f"vllm_dp{i}",
                    "type": "VLLM",
                    "is_target_for_workload": True,
                    "config": {
                        "model_profile_id": "Llama2-7B",
                        "gpu_id": f"gpu{i}",
                        "block_size": 16,
                        "max_num_seqs": 16,  # Limited per instance
                        "max_num_batched_tokens": 1024,
                        "scheduler_iteration_delay_s": 0.0001
                    }
                })
            
            config["metrics_config"]["output_summary_json_path"] = str(tmp_path / f"dp_{strategy}_summary.json")
            config["metrics_config"]["output_requests_csv_path"] = str(tmp_path / f"dp_{strategy}_requests.csv")
            
            orchestrator = ExperimentOrchestrator(config)
            orchestrator.run()
            
            with open(tmp_path / f"dp_{strategy}_summary.json") as f:
                results[strategy] = json.load(f)
        
        # All strategies should achieve reasonable success rates
        for strategy in strategies:
            assert results[strategy]["requests"]["success_rate"] > 0.8
        
        # Least loaded should have better latency consistency
        assert results["least_loaded"]["latency"]["time_to_first_token_ms"]["std"] <= \
               results["random"]["latency"]["time_to_first_token_ms"]["std"]
    
    def test_combined_tp_pp(self, base_config, tmp_path):
        """Test combined tensor and pipeline parallelism."""
        config = base_config.copy()
        config["frameworks_to_test"] = [{
            "name": "parallel_vllm_tp2_pp2",
            "type": "ParallelVLLM",
            "is_target_for_workload": True,
            "config": {
                "model_profile_id": "Llama2-7B",
                "gpu_id": "gpu0",
                "block_size": 16,
                "max_num_seqs": 64,
                "parallelism": {
                    "strategy": "TP_PP",
                    "tp_degree": 2,
                    "pp_stages": 2,
                    "num_microbatches_per_request": 4,
                    "gpu_ids": ["gpu0", "gpu1", "gpu2", "gpu3"]
                }
            }
        }]
        
        config["metrics_config"]["output_summary_json_path"] = str(tmp_path / "tp_pp_summary.json")
        config["metrics_config"]["output_requests_csv_path"] = str(tmp_path / "tp_pp_requests.csv")
        
        orchestrator = ExperimentOrchestrator(config)
        orchestrator.run()
        
        with open(tmp_path / "tp_pp_summary.json") as f:
            results = json.load(f)
        
        # Combined parallelism should work
        assert results["requests"]["success_rate"] > 0.9
        assert results["throughput"]["output_tokens_per_second"] > 0
    
    def test_large_model_parallelism(self, base_config, tmp_path):
        """Test parallelism with larger models that require it."""
        config = base_config.copy()
        
        # Use a larger model that needs parallelism
        config["frameworks_to_test"] = [{
            "name": "vllm_large_tp4",
            "type": "VLLM",
            "is_target_for_workload": True,
            "config": {
                "model_profile_id": "GPT-3-175B",  # Large model
                "gpu_id": "gpu0",
                "block_size": 16,
                "max_num_seqs": 8,  # Fewer sequences due to memory
                "max_num_batched_tokens": 512,
                "scheduler_iteration_delay_s": 0.0001,
                "parallelism": {
                    "strategy": "TP",
                    "tp_degree": 4,
                    "gpu_ids": ["gpu0", "gpu1", "gpu2", "gpu3"]
                }
            }
        }]
        
        # Adjust workload for larger model
        config["workload"]["client_profiles"][0]["inter_arrival_time_dist_config"]["rate"] = 0.2  # Slower
        config["workload"]["client_profiles"][0]["prompt_tokens_dist_config"] = {
            "type": "Constant",
            "value": 50,
            "is_int": True
        }
        config["workload"]["client_profiles"][0]["max_output_tokens_dist_config"]["value"] = 50
        
        config["metrics_config"]["output_summary_json_path"] = str(tmp_path / "large_model_summary.json")
        config["metrics_config"]["output_requests_csv_path"] = str(tmp_path / "large_model_requests.csv")
        
        orchestrator = ExperimentOrchestrator(config)
        orchestrator.run()
        
        with open(tmp_path / "large_model_summary.json") as f:
            results = json.load(f)
        
        # Large model should still handle some requests
        assert results["requests"]["total"] > 0
        # But latency will be higher
        assert results["latency"]["time_to_first_token_ms"]["mean"] > 100  # Expect higher latency


class TestParallelismPerformanceValidation:
    """Validate that parallelism provides expected performance benefits."""
    
    def test_tp_reduces_prefill_latency(self, tmp_path):
        """Verify TP reduces prefill latency for compute-bound operations."""
        base_config = {
            "simulation": {"max_simulation_time": 20, "random_seed": 42},
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
                        "link_id": f"gpu{i}_to_gpu{j}",
                        "source_id": f"gpu{i}",
                        "dest_id": f"gpu{j}",
                        "bandwidth_bps": 600_000_000_000,
                        "latency_s": 0.0000005,
                        "bidirectional": True
                    }
                    for i in range(4) for j in range(i+1, 4)
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
                "total_duration": 20,
                "bytes_per_token_estimate_for_network": 2,
                "client_profiles": [{
                    "profile_name": "large_prompts",
                    "weight": 1.0,
                    "inter_arrival_time_dist_config": {"type": "Constant", "value": 2.0},
                    "prompt_tokens_dist_config": {"type": "Constant", "value": 1000, "is_int": True},
                    "max_output_tokens_dist_config": {"type": "Constant", "value": 10, "is_int": True},
                    "conversational_probability": 0.0,
                    "streaming_response_probability": 0.0
                }]
            },
            "metrics_config": {
                "percentiles_to_calculate": [0.5, 0.9, 0.99],
                "warm_up_duration_s": 2
            }
        }
        
        # Test single GPU
        single_config = base_config.copy()
        single_config["frameworks_to_test"] = [{
            "name": "vllm_single",
            "type": "VLLM",
            "is_target_for_workload": True,
            "config": {
                "model_profile_id": "Llama2-7B",
                "gpu_id": "gpu0",
                "block_size": 16,
                "max_num_seqs": 32,
                "max_num_batched_tokens": 2048
            }
        }]
        single_config["metrics_config"]["output_summary_json_path"] = str(tmp_path / "single_ttft.json")
        single_config["metrics_config"]["output_requests_csv_path"] = str(tmp_path / "single_ttft.csv")
        
        orchestrator = ExperimentOrchestrator(single_config)
        orchestrator.run()
        
        with open(tmp_path / "single_ttft.json") as f:
            single_results = json.load(f)
        
        # Test TP=4
        tp4_config = base_config.copy()
        tp4_config["frameworks_to_test"] = [{
            "name": "vllm_tp4",
            "type": "VLLM",
            "is_target_for_workload": True,
            "config": {
                "model_profile_id": "Llama2-7B",
                "gpu_id": "gpu0",
                "block_size": 16,
                "max_num_seqs": 32,
                "max_num_batched_tokens": 2048,
                "parallelism": {
                    "strategy": "TP",
                    "tp_degree": 4,
                    "gpu_ids": ["gpu0", "gpu1", "gpu2", "gpu3"]
                }
            }
        }]
        tp4_config["metrics_config"]["output_summary_json_path"] = str(tmp_path / "tp4_ttft.json")
        tp4_config["metrics_config"]["output_requests_csv_path"] = str(tmp_path / "tp4_ttft.csv")
        
        orchestrator = ExperimentOrchestrator(tp4_config)
        orchestrator.run()
        
        with open(tmp_path / "tp4_ttft.json") as f:
            tp4_results = json.load(f)
        
        # TP should reduce TTFT for large prompts (compute-bound)
        single_ttft = single_results["latency"]["time_to_first_token_ms"]["p50"]
        tp4_ttft = tp4_results["latency"]["time_to_first_token_ms"]["p50"]
        
        # Expect significant reduction (accounting for communication overhead)
        assert tp4_ttft < single_ttft * 0.4  # Should be faster than 40% of single GPU
        
        # But TPOT should be similar (memory-bound)
        single_tpot = single_results["latency"]["time_per_output_token_ms"]["p50"]
        tp4_tpot = tp4_results["latency"]["time_per_output_token_ms"]["p50"]
        assert abs(tp4_tpot - single_tpot) / single_tpot < 0.2  # Within 20%