"""Basic integration test for the simulation system."""

import json
import tempfile
from pathlib import Path

import pytest

from llmperforacle.orchestration import ExperimentOrchestrator


def test_basic_simulation():
    """Test that a basic simulation can run end-to-end."""
    # Create minimal configuration
    config = {
        "simulation": {
            "max_simulation_time": 10,  # Short simulation
            "random_seed": 42,
        },
        "model_characteristics_db_path": "configs/model_params.json",
        "hardware_profile": {
            "compute_devices": [
                {
                    "device_id": "gpu0",
                    "device_type": "GPU",
                    "peak_tflops": {"fp16": 100},
                    "memory_capacity_bytes": 16_000_000_000,
                    "memory_gbps": 900,
                    "processing_units": 80,
                }
            ],
            "network_links": [
                {
                    "link_id": "client_link",
                    "source_id": "client_node_0",
                    "dest_id": "gpu0",
                    "bandwidth_bps": 1_000_000_000,
                    "latency_s": 0.001,
                }
            ],
        },
        "workload": {
            "total_duration": 10,
            "bytes_per_token_estimate_for_network": 2,
            "random_seed": 123,
            "client_profiles": [
                {
                    "profile_name": "test_client",
                    "weight": 1.0,
                    "inter_arrival_time_dist_config": {
                        "type": "Exponential",
                        "rate": 2.0,  # 2 requests per second average
                    },
                    "prompt_tokens_dist_config": {
                        "type": "Constant",
                        "value": 50,
                        "is_int": True,
                    },
                    "max_output_tokens_dist_config": {
                        "type": "Constant",
                        "value": 100,
                        "is_int": True,
                    },
                    "conversational_probability": 0.0,
                    "streaming_response_probability": 1.0,
                }
            ],
        },
        "frameworks_to_test": [
            {
                "name": "test_vllm",
                "type": "VLLM",
                "is_target_for_workload": True,
                "config": {
                    "model_profile_id": "Llama2-7B",
                    "gpu_id": "gpu0",
                    "block_size": 16,
                    "max_num_seqs": 10,
                    "max_num_batched_tokens": 512,
                    "scheduler_iteration_delay_s": 0.001,
                    "bytes_per_token_estimate_for_network": 2,
                },
            }
        ],
        "metrics_config": {
            "percentiles_to_calculate": [0.5, 0.9, 0.99],
            "warm_up_duration_s": 2,
        },
    }
    
    # Create orchestrator and run simulation
    orchestrator = ExperimentOrchestrator(config)
    summary = orchestrator.run()
    
    # Verify basic results
    assert summary is not None
    assert "simulation" in summary
    assert "requests" in summary
    assert "throughput" in summary
    assert "latency" in summary
    
    # Check that some requests were processed
    assert summary["requests"]["total"] > 0
    assert summary["throughput"]["output_tokens_per_second"] > 0


if __name__ == "__main__":
    test_basic_simulation()
    print("Basic simulation test passed!")