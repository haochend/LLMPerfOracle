"""Debug test for DP load balancing."""

import json
import logging
import tempfile
from pathlib import Path

from llmperforacle.orchestration import ExperimentOrchestrator

# Enable logging to see load balancing decisions
logging.basicConfig(level=logging.INFO)


def test_dp_load_distribution():
    """Test that requests are actually distributed across DP replicas."""
    
    config = {
        "simulation": {"max_simulation_time": 10, "random_seed": 42},
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
                for i in range(2)
            ],
            "network_links": [
                {
                    "link_id": "gpu0_to_gpu1",
                    "source_id": "gpu0",
                    "dest_id": "gpu1", 
                    "bandwidth_bps": 600_000_000_000,
                    "latency_s": 0.0000005,
                    "bidirectional": True
                },
                {
                    "link_id": "client_to_server",
                    "source_id": "client_node_0",
                    "dest_id": "framework_entry_0",
                    "bandwidth_bps": 10_000_000_000,
                    "latency_s": 0.0001,
                    "bidirectional": True
                }
            ]
        },
        "workload": {
            "total_duration": 10,
            "bytes_per_token_estimate_for_network": 2,
            "load_balancing_strategy": "round_robin",
            "client_profiles": [{
                "profile_name": "test",
                "weight": 1.0,
                "inter_arrival_time_dist_config": {
                    "type": "Exponential",
                    "rate": 10.0  # 10 req/s
                },
                "prompt_tokens_dist_config": {
                    "type": "Constant",
                    "value": 100,
                    "is_int": True
                },
                "max_output_tokens_dist_config": {
                    "type": "Constant", 
                    "value": 50,
                    "is_int": True
                },
                "conversational_probability": 0.0,
                "streaming_response_probability": 0.0
            }]
        },
        "frameworks_to_test": [
            {
                "name": "vllm_dp0",
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
            },
            {
                "name": "vllm_dp1", 
                "type": "VLLM",
                "is_target_for_workload": True,
                "config": {
                    "model_profile_id": "Llama2-7B",
                    "gpu_id": "gpu1",
                    "block_size": 16,
                    "max_num_seqs": 32,
                    "max_num_batched_tokens": 2048,
                    "scheduler_iteration_delay_s": 0.0001
                }
            }
        ],
        "metrics_config": {
            "percentiles_to_calculate": [0.5, 0.9],
            "warm_up_duration_s": 1
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        config["metrics_config"]["output_summary_json_path"] = str(tmpdir / "dp_test.json")
        config["metrics_config"]["output_requests_csv_path"] = str(tmpdir / "dp_test.csv")
        
        # Run simulation
        orchestrator = ExperimentOrchestrator(config)
        orchestrator.run()
        
        # Check results
        with open(tmpdir / "dp_test.json") as f:
            results = json.load(f)
        
        # Read CSV to check request distribution
        import csv
        with open(tmpdir / "dp_test.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Count requests per framework
        fw_counts = {"vllm_dp0": 0, "vllm_dp1": 0}
        for row in rows:
            # The CSV should have a framework_id or similar field
            # Let's print the first row to see the structure
            if fw_counts["vllm_dp0"] == 0 and fw_counts["vllm_dp1"] == 0:
                print(f"CSV columns: {list(row.keys())}")
            
            # Look for framework assignment - might be in different columns
            for key, value in row.items():
                if "framework" in key.lower() and value in fw_counts:
                    fw_counts[value] += 1
                    break
        
        print(f"\nResults:")
        print(f"Total requests: {results['requests']['total']}")
        print(f"Success rate: {results['requests']['success_rate']:.1%}")
        print(f"Throughput: {results['throughput']['requests_per_second']:.1f} req/s")
        
        # Check per-framework metrics if available
        if 'per_framework' in results:
            print(f"\nPer-framework results:")
            for fw_id, fw_metrics in results['per_framework'].items():
                print(f"  {fw_id}: {fw_metrics.get('requests', {}).get('total', 0)} requests")
        
        # Let's also add a direct test of the workload generator
        print("\nDirect workload generator test:")
        test_load_balancing_directly()


def test_load_balancing_directly():
    """Test the workload generator's load balancing directly."""
    import simpy
    from llmperforacle.workload import WorkloadGenerator
    
    env = simpy.Environment()
    
    # Create mock target frameworks
    targets = {
        "fw0": simpy.Store(env),
        "fw1": simpy.Store(env),
        "fw2": simpy.Store(env)
    }
    
    config = {
        "total_duration": 5,
        "load_balancing_strategy": "round_robin",
        "client_profiles": [{
            "profile_name": "test",
            "weight": 1.0,
            "inter_arrival_time_dist_config": {"type": "Constant", "value": 0.1},
            "prompt_tokens_dist_config": {"type": "Constant", "value": 100},
            "max_output_tokens_dist_config": {"type": "Constant", "value": 50},
            "conversational_probability": 0.0
        }]
    }
    
    wg = WorkloadGenerator(
        simpy_env=env,
        config=config,
        target_frameworks_map=targets,
        metrics_collector=None,
        hardware_platform=None
    )
    
    # Check framework request counts after initialization
    print(f"Framework request counts: {wg.framework_request_counts}")
    
    # Simulate some request dispatching
    from llmperforacle.workload.models import Request
    for i in range(12):
        # Increment counter as would happen in normal flow
        wg.request_counter += 1
        req = Request(
            request_id=f"test_{i}",
            client_id="client",
            session_id="sess",
            arrival_time=i * 0.1,
            prompt_num_tokens=100,
            max_output_tokens=50,
            is_conversational_turn=False,
            streaming_response=False,
            user_priority=0
        )
        target_fw = wg._select_target_framework(req)
        print(f"Request {i} (counter={wg.request_counter}) -> {target_fw}")
    
    print(f"\nFinal distribution: {wg.framework_request_counts}")


if __name__ == "__main__":
    test_dp_load_distribution()