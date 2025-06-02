"""
Simple demonstration of LoD performance improvements.
"""

import time
import pytest
from llmperforacle.orchestration import ExperimentOrchestrator


def create_simple_config(lod="high", duration=30):
    """Create a simple test configuration."""
    return {
        "simulation": {
            "max_simulation_time": duration,
            "random_seed": 42,
            "lod": lod
        },
        "model_characteristics_db_path": "./configs/model_params.json",
        "hardware_profile": {
            "compute_devices": [
                {
                    "device_id": "gpu0",
                    "device_type": "GPU",
                    "peak_tflops": {"fp16": 312, "int8": 624},
                    "memory_capacity_bytes": 80_000_000_000,
                    "memory_gbps": 2039,
                    "processing_units": 108
                }
            ],
            "network_links": [
                {
                    "link_id": "client_to_server",
                    "source_id": "client_node_0",
                    "dest_id": "gpu0",
                    "bandwidth_bps": 10_000_000_000,
                    "latency_s": 0.0001,
                    "bidirectional": True
                }
            ]
        },
        "workload": {
            "total_duration": duration,
            "bytes_per_token_estimate_for_network": 2,
            "random_seed": 123,
            "max_turns_per_session": 1,
            "client_profiles": [
                {
                    "profile_name": "test_load",
                    "weight": 1.0,
                    "inter_arrival_time_dist_config": {
                        "type": "Exponential",
                        "rate": 1.0  # 1 request per second
                    },
                    "prompt_tokens_dist_config": {
                        "type": "Fixed",
                        "value": 256
                    },
                    "max_output_tokens_dist_config": {
                        "type": "Fixed", 
                        "value": 128
                    },
                    "turn_type_dist_config": {
                        "type": "Fixed",
                        "value": "new_conversation"
                    }
                }
            ]
        },
        "frameworks_to_test": [
            {
                "name": "vllm_test",
                "type": "VLLM",
                "is_target_for_workload": True,
                "config": {
                    "model_profile_id": "Llama2-7B",  # Smaller model
                    "gpu_id": "gpu0",
                    "block_size": 16,
                    "max_num_seqs": 256,
                    "enable_prefix_caching": False,
                    "enable_cross_request_caching": False,
                    "enable_chunked_prefill": True,
                    "prefill_chunk_size": 2048
                }
            }
        ],
        "metrics_config": {
            "output_summary_json_path": "./experiments/results/lod_simple_demo.json",
            "output_requests_csv_path": "./experiments/results/lod_simple_demo.csv",
            "compute_token_stats": True,
            "compute_percentiles": [50, 90, 95, 99]
        }
    }


class TestLoDSimpleDemo:
    """Simple LoD demonstration tests."""
    
    def test_lod_speedup_basic(self):
        """Test basic speedup with LoD abstractions."""
        # Create configs
        high_config = create_simple_config(lod="high", duration=20)
        medium_config = create_simple_config(lod="medium", duration=20)
        
        # Run high LoD
        start_time = time.time()
        high_orchestrator = ExperimentOrchestrator(high_config)
        high_report = high_orchestrator.run()
        high_time = time.time() - start_time
        
        # Run medium LoD
        start_time = time.time()
        medium_orchestrator = ExperimentOrchestrator(medium_config)
        medium_report = medium_orchestrator.run()
        medium_time = time.time() - start_time
        
        # Calculate speedup
        speedup = high_time / medium_time
        
        print(f"\n{'='*60}")
        print(f"LoD Performance Comparison - Basic Test")
        print(f"{'='*60}")
        print(f"High LoD wall clock time: {high_time:.2f}s")
        print(f"Medium LoD wall clock time: {medium_time:.2f}s")
        print(f"Speedup factor: {speedup:.2f}x")
        
        # Extract metrics
        high_requests = high_report['requests']['successful']
        medium_requests = medium_report['requests']['successful']
        
        print(f"\nCompleted requests:")
        print(f"  High LoD: {high_requests}")
        print(f"  Medium LoD: {medium_requests}")
        
        # Check that medium LoD is faster
        assert speedup > 1.0, f"Expected speedup > 1.0, got {speedup}"
        
        # Check that metrics are similar (within 10%)
        if high_requests > 0:
            diff_pct = abs(medium_requests - high_requests) / high_requests * 100
            print(f"  Difference: {diff_pct:.1f}%")
            assert diff_pct < 20, f"Request count differs by {diff_pct:.1f}%"
        
        print(f"{'='*60}\n")
    
    def test_lod_with_tensor_parallelism(self):
        """Test LoD speedup with tensor parallelism."""
        config_base = create_simple_config(duration=20)
        
        # Add more GPUs
        for i in range(1, 4):
            config_base["hardware_profile"]["compute_devices"].append({
                "device_id": f"gpu{i}",
                "device_type": "GPU",
                "peak_tflops": {"fp16": 312, "int8": 624},
                "memory_capacity_bytes": 80_000_000_000,
                "memory_gbps": 2039,
                "processing_units": 108
            })
        
        # Add inter-GPU links
        for i in range(4):
            config_base["hardware_profile"]["network_links"].append({
                "link_id": f"gpu{i}_to_gpu{(i+1)%4}",
                "source_id": f"gpu{i}",
                "dest_id": f"gpu{(i+1)%4}",
                "bandwidth_bps": 600_000_000_000,  # NVLink
                "latency_s": 0.000001,
                "bidirectional": True
            })
        
        # Update framework to use TP
        config_base["frameworks_to_test"][0] = {
            "name": "vllm_tp4",
            "type": "ParallelVLLM",
            "is_target_for_workload": True,
            "config": {
                "model_profile_id": "Llama2-13B",
                "block_size": 16,
                "max_num_seqs": 256,
                "enable_prefix_caching": False,
                "enable_cross_request_caching": False,
                "enable_chunked_prefill": True,
                "prefill_chunk_size": 4096,
                "parallelism": {
                    "strategy": "TP",
                    "tp_degree": 4,
                    "gpu_ids": ["gpu0", "gpu1", "gpu2", "gpu3"]
                }
            }
        }
        
        # Create high and medium configs
        high_config = config_base.copy()
        high_config["simulation"]["lod"] = "high"
        
        medium_config = config_base.copy() 
        medium_config["simulation"]["lod"] = "medium"
        
        # Run tests
        start_time = time.time()
        high_orchestrator = ExperimentOrchestrator(high_config)
        high_report = high_orchestrator.run()
        high_time = time.time() - start_time
        
        start_time = time.time()
        medium_orchestrator = ExperimentOrchestrator(medium_config)
        medium_report = medium_orchestrator.run()
        medium_time = time.time() - start_time
        
        speedup = high_time / medium_time
        
        print(f"\n{'='*60}")
        print(f"LoD Performance Comparison - Tensor Parallelism")
        print(f"{'='*60}")
        print(f"High LoD wall clock time: {high_time:.2f}s")
        print(f"Medium LoD wall clock time: {medium_time:.2f}s")
        print(f"Speedup factor: {speedup:.2f}x")
        print(f"{'='*60}\n")
        
        assert speedup > 1.0, f"Expected speedup > 1.0 for TP, got {speedup}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])