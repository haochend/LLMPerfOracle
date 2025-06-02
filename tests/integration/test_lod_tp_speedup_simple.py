"""
Simple test to demonstrate LoD speedup with Tensor Parallelism.
"""

import time
import pytest
from llmperforacle.orchestration import ExperimentOrchestrator


def create_tp_test_config(tp_degree=4, lod="high", duration=30):
    """Create a simple TP test configuration."""
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
                    "device_id": f"gpu{i}",
                    "device_type": "GPU", 
                    "peak_tflops": {"fp16": 312, "int8": 624},
                    "memory_capacity_bytes": 80_000_000_000,
                    "memory_gbps": 2039,
                    "processing_units": 108
                }
                for i in range(tp_degree)
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
            ] + [
                # Create all-to-all links for GPUs
                {
                    "link_id": f"gpu{i}_to_gpu{j}",
                    "source_id": f"gpu{i}",
                    "dest_id": f"gpu{j}",
                    "bandwidth_bps": 600_000_000_000,  # 600 Gbps NVLink
                    "latency_s": 0.000001,
                    "bidirectional": True
                }
                for i in range(tp_degree)
                for j in range(tp_degree)
                if i != j
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
                        "rate": 2.0 * tp_degree  # Scale with TP degree
                    },
                    "prompt_tokens_dist_config": {
                        "type": "Fixed",
                        "value": 512
                    },
                    "max_output_tokens_dist_config": {
                        "type": "Fixed",
                        "value": 128
                    }
                }
            ]
        },
        "frameworks_to_test": [
            {
                "name": f"vllm_tp{tp_degree}",
                "type": "ParallelVLLM",
                "is_target_for_workload": True,
                "config": {
                    "model_profile_id": "Llama3-8B",  # 32 layers
                    "block_size": 16,
                    "max_num_seqs": 256,
                    "enable_prefix_caching": False,
                    "enable_cross_request_caching": False,
                    "enable_chunked_prefill": True,
                    "prefill_chunk_size": 2048,
                    "parallelism": {
                        "strategy": "TP",
                        "tp_degree": tp_degree,
                        "gpu_ids": [f"gpu{i}" for i in range(tp_degree)]
                    }
                }
            }
        ],
        "metrics_config": {
            "output_summary_json_path": f"./experiments/results/tp{tp_degree}_{lod}_lod.json",
            "output_requests_csv_path": f"./experiments/results/tp{tp_degree}_{lod}_lod.csv",
            "compute_token_stats": True,
            "compute_percentiles": [50, 90, 95, 99]
        }
    }


class TestLoDTPSpeedupSimple:
    """Simple tests for TP LoD speedup."""
    
    @pytest.mark.parametrize("tp_degree", [2, 4])
    def test_tp_lod_speedup(self, tp_degree):
        """Test LoD speedup with different TP degrees."""
        duration = 20  # Short test
        
        # Run high LoD
        high_config = create_tp_test_config(tp_degree, "high", duration)
        start_time = time.time()
        high_orch = ExperimentOrchestrator(high_config)
        high_report = high_orch.run()
        high_time = time.time() - start_time
        
        # Run medium LoD
        medium_config = create_tp_test_config(tp_degree, "medium", duration)
        start_time = time.time()
        medium_orch = ExperimentOrchestrator(medium_config)
        medium_report = medium_orch.run()
        medium_time = time.time() - start_time
        
        speedup = high_time / medium_time
        
        print(f"\n{'='*60}")
        print(f"TP{tp_degree} LoD Speedup Test Results")
        print(f"{'='*60}")
        print(f"High LoD time:   {high_time:.2f}s")
        print(f"Medium LoD time: {medium_time:.2f}s")
        print(f"Speedup:         {speedup:.2f}x")
        
        # Check request counts
        high_requests = high_report['requests']['successful']
        medium_requests = medium_report['requests']['successful']
        
        print(f"\nCompleted requests:")
        print(f"  High LoD:   {high_requests}")
        print(f"  Medium LoD: {medium_requests}")
        
        # Calculate event reduction
        # With 32 layers, TP operations generate many collective events
        # High LoD: 32 layers * 2 (attn+mlp) * 2 (compute+collective) = 128 events per step
        # Medium LoD: 1 compute + 1 collective = 2 events per step
        event_reduction = 128 / 2
        print(f"\nTheoretical event reduction: {event_reduction:.0f}x")
        print(f"{'='*60}\n")
        
        # Assertions
        assert speedup > 1.2, f"Expected speedup > 1.2x for TP{tp_degree}, got {speedup:.2f}x"
        
        # Check accuracy
        if high_requests > 0:
            diff_pct = abs(medium_requests - high_requests) / high_requests * 100
            assert diff_pct < 20, f"Request difference {diff_pct:.1f}% exceeds 20%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])