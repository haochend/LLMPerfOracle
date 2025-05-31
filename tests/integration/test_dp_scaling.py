"""Clean test for DP scaling to debug throughput issue."""

import json
import tempfile
from pathlib import Path

from llmperforacle.orchestration import ExperimentOrchestrator


def test_dp_scaling():
    """Test that DP scales throughput properly."""
    
    def run_dp_test(num_replicas, tmp_path):
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
                    for i in range(num_replicas)
                ],
                "network_links": []
            },
            "workload": {
                "total_duration": 30,
                "bytes_per_token_estimate_for_network": 2,
                "load_balancing_strategy": "round_robin",
                "client_profiles": [{
                    "profile_name": "high_throughput",
                    "weight": 1.0,
                    "inter_arrival_time_dist_config": {
                        "type": "Exponential",
                        "rate": 30.0  # 30 req/s - should be in saturation zone
                    },
                    "prompt_tokens_dist_config": {
                        "type": "Uniform",
                        "low": 300,
                        "high": 500,
                        "is_int": True
                    },
                    "max_output_tokens_dist_config": {
                        "type": "Uniform",
                        "low": 300,
                        "high": 500,
                        "is_int": True
                    },
                    "conversational_probability": 0.0,
                    "streaming_response_probability": 0.0
                }]
            },
            "frameworks_to_test": [],
            "metrics_config": {
                "percentiles_to_calculate": [0.5, 0.9],
                "warm_up_duration_s": 5,
                "output_summary_json_path": str(tmp_path / f"dp{num_replicas}_clean.json"),
                "output_requests_csv_path": str(tmp_path / f"dp{num_replicas}_clean.csv")
            }
        }
        
        # Add network links
        for i in range(num_replicas):
            for j in range(i + 1, num_replicas):
                config["hardware_profile"]["network_links"].append({
                    "link_id": f"gpu{i}_to_gpu{j}",
                    "source_id": f"gpu{i}",
                    "dest_id": f"gpu{j}",
                    "bandwidth_bps": 600_000_000_000,
                    "latency_s": 0.0000005,
                    "bidirectional": True
                })
        
        # Add client network link
        config["hardware_profile"]["network_links"].append({
            "link_id": "client_to_server",
            "source_id": "client_node_0",
            "dest_id": "framework_entry_0",
            "bandwidth_bps": 10_000_000_000,
            "latency_s": 0.0001,
            "bidirectional": True
        })
        
        # Add framework instances
        for i in range(num_replicas):
            config["frameworks_to_test"].append({
                "name": f"vllm_dp{i}",
                "type": "VLLM",
                "is_target_for_workload": True,
                "config": {
                    "model_profile_id": "Llama2-7B",
                    "gpu_id": f"gpu{i}",
                    "block_size": 16,
                    "max_num_seqs": 32,
                    "max_num_batched_tokens": 2048,
                    "scheduler_iteration_delay_s": 0.0001
                }
            })
        
        # Run simulation
        orchestrator = ExperimentOrchestrator(config)
        orchestrator.run()
        
        # Load results
        with open(tmp_path / f"dp{num_replicas}_clean.json") as f:
            return json.load(f)
    
    # Test with 1 and 2 replicas
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        print("\nRunning DP=1 test...")
        results_1 = run_dp_test(1, tmpdir)
        print("\nRunning DP=2 test...")
        results_2 = run_dp_test(2, tmpdir)
        
        # Compare results
        print("\n=== DP Scaling Results ===")
        print(f"\nDP=1:")
        print(f"  Total requests: {results_1['requests']['total']}")
        print(f"  Successful requests: {results_1['requests']['successful']}")
        print(f"  Success rate: {results_1['requests']['success_rate']:.1%}")
        print(f"  Throughput: {results_1['throughput']['requests_per_second']:.1f} req/s")
        
        print(f"\nDP=2:")
        print(f"  Total requests: {results_2['requests']['total']}")
        print(f"  Successful requests: {results_2['requests']['successful']}")
        print(f"  Success rate: {results_2['requests']['success_rate']:.1%}")
        print(f"  Throughput: {results_2['throughput']['requests_per_second']:.1f} req/s")
        
        # Check per-framework metrics
        if 'per_framework' in results_2:
            print("\nPer-framework request distribution (DP=2):")
            for fw_id, fw_metrics in results_2['per_framework'].items():
                print(f"  {fw_id}: {fw_metrics.get('requests', {}).get('total', 0)} requests")
        
        # Calculate scaling
        throughput_1 = results_1['throughput']['requests_per_second']
        throughput_2 = results_2['throughput']['requests_per_second']
        
        if throughput_1 > 0:
            scaling = throughput_2 / throughput_1
            print(f"\nThroughput scaling: {scaling:.2f}x (expected: ~1.8-2.0x)")
        else:
            print(f"\nDP=1 had 0 throughput, cannot calculate scaling ratio")
        
        # Also check CSV for request distribution
        import csv
        with open(tmpdir / "dp2_clean.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                print(f"\nCSV columns: {list(rows[0].keys())}")
                # Count by framework if possible
                fw_counts = {}
                for row in rows:
                    # Look for framework field
                    for key in ['framework_id', 'framework', 'target_framework']:
                        if key in row and row[key]:
                            fw = row[key]
                            fw_counts[fw] = fw_counts.get(fw, 0) + 1
                            break
                if fw_counts:
                    print("\nRequest distribution from CSV:")
                    for fw, count in sorted(fw_counts.items()):
                        print(f"  {fw}: {count} requests")


if __name__ == "__main__":
    test_dp_scaling()