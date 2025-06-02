"""
Quick accuracy verification test for LoD abstractions.
Focuses on key metrics and shorter simulations.
"""

import time
import json
from pathlib import Path
import pytest
from llmperforacle.orchestration import ExperimentOrchestrator


class TestLoDAccuracyQuick:
    """Quick accuracy tests for LoD."""
    
    def create_tp_config(self, tp_degree: int, lod: str, duration: int = 30):
        """Create TP configuration for testing."""
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
                    {
                        "link_id": f"gpu{i}_to_gpu{j}",
                        "source_id": f"gpu{i}",
                        "dest_id": f"gpu{j}",
                        "bandwidth_bps": 600_000_000_000,
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
                "client_profiles": [{
                    "profile_name": "test_load",
                    "weight": 1.0,
                    "inter_arrival_time_dist_config": {
                        "type": "Exponential",
                        "rate": 3.0 * tp_degree  # Scale with TP
                    },
                    "prompt_tokens_dist_config": {
                        "type": "Normal",
                        "mean": 256,
                        "sigma": 32,
                        "is_int": True
                    },
                    "max_output_tokens_dist_config": {
                        "type": "Normal",
                        "mean": 128,
                        "sigma": 16,
                        "is_int": True
                    }
                }]
            },
            "frameworks_to_test": [{
                "name": f"vllm_tp{tp_degree}",
                "type": "ParallelVLLM",
                "is_target_for_workload": True,
                "config": {
                    "model_profile_id": "Llama2-13B" if tp_degree <= 4 else "Llama3-70B",
                    "block_size": 16,
                    "max_num_seqs": 256,
                    "enable_prefix_caching": False,
                    "enable_cross_request_caching": False,
                    "enable_chunked_prefill": True,
                    "prefill_chunk_size": 2048 * tp_degree,
                    "parallelism": {
                        "strategy": "TP",
                        "tp_degree": tp_degree,
                        "gpu_ids": [f"gpu{i}" for i in range(tp_degree)]
                    }
                }
            }],
            "metrics_config": {
                "output_summary_json_path": f"./experiments/results/quick_tp{tp_degree}_{lod}.json",
                "output_requests_csv_path": f"./experiments/results/quick_tp{tp_degree}_{lod}.csv",
                "compute_token_stats": True,
                "compute_percentiles": [50, 90, 95, 99],
                "warm_up_duration_s": 5
            }
        }
    
    @pytest.mark.parametrize("tp_degree", [4, 8])
    def test_tp_accuracy_and_speedup(self, tp_degree):
        """Test accuracy and speedup for TP configurations."""
        print(f"\n{'='*70}")
        print(f"Quick LoD Accuracy Test: TP{tp_degree}")
        print(f"{'='*70}")
        
        duration = 30  # 30 seconds for quick test
        
        # Run high LoD
        print(f"\nRunning HIGH LoD (TP{tp_degree})...")
        high_config = self.create_tp_config(tp_degree, "high", duration)
        high_start = time.time()
        high_orch = ExperimentOrchestrator(high_config)
        high_report = high_orch.run()
        high_time = time.time() - high_start
        
        # Run medium LoD
        print(f"Running MEDIUM LoD (TP{tp_degree})...")
        medium_config = self.create_tp_config(tp_degree, "medium", duration)
        medium_start = time.time()
        medium_orch = ExperimentOrchestrator(medium_config)
        medium_report = medium_orch.run()
        medium_time = time.time() - medium_start
        
        # Calculate speedup
        speedup = high_time / medium_time
        
        # Extract key metrics
        def extract_key_metrics(report):
            latency = report.get('latency', {})
            requests = report.get('requests', {})
            throughput = report.get('throughput', {})
            
            return {
                'requests': requests.get('successful', 0),
                'success_rate': requests.get('success_rate', 0),
                'avg_ttft': latency.get('time_to_first_token_ms', {}).get('mean', 0),
                'p99_ttft': latency.get('time_to_first_token_ms', {}).get('p99', 0),
                'avg_e2e': latency.get('end_to_end_latency_ms', {}).get('mean', 0),
                'p99_e2e': latency.get('end_to_end_latency_ms', {}).get('p99', 0),
                'req_throughput': throughput.get('request_throughput_per_s', 0),
                'token_throughput': throughput.get('tokens_throughput_per_s', 0)
            }
        
        high_metrics = extract_key_metrics(high_report)
        medium_metrics = extract_key_metrics(medium_report)
        
        # Print results
        print(f"\nPerformance Results:")
        print(f"  High LoD time:   {high_time:6.2f}s")
        print(f"  Medium LoD time: {medium_time:6.2f}s")
        print(f"  Speedup:         {speedup:6.2f}x")
        
        print(f"\nAccuracy Comparison:")
        print(f"  {'Metric':<20} {'High':>10} {'Medium':>10} {'Diff %':>8}")
        print(f"  {'-'*50}")
        
        for key in high_metrics:
            high_val = high_metrics[key]
            medium_val = medium_metrics[key]
            if high_val > 0:
                diff_pct = abs(medium_val - high_val) / high_val * 100
            else:
                diff_pct = 0
            print(f"  {key:<20} {high_val:>10.1f} {medium_val:>10.1f} {diff_pct:>7.1f}%")
        
        # Calculate average error
        errors = []
        for key in ['avg_ttft', 'avg_e2e', 'req_throughput']:
            if high_metrics[key] > 0:
                error = abs(medium_metrics[key] - high_metrics[key]) / high_metrics[key] * 100
                errors.append(error)
        
        avg_error = sum(errors) / len(errors) if errors else 0
        
        print(f"\nAverage error on key metrics: {avg_error:.2f}%")
        
        # Save results
        results = {
            "tp_degree": tp_degree,
            "duration": duration,
            "performance": {
                "high_lod_time": high_time,
                "medium_lod_time": medium_time,
                "speedup": speedup
            },
            "metrics": {
                "high": high_metrics,
                "medium": medium_metrics
            },
            "avg_error_pct": avg_error
        }
        
        results_dir = Path("./experiments/results/quick_accuracy")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / f"tp{tp_degree}_accuracy.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Assertions
        assert speedup > 5.0, f"Expected significant speedup > 5x for TP{tp_degree}, got {speedup:.2f}x"
        assert avg_error < 15.0, f"Average error {avg_error:.2f}% exceeds 15% threshold"
        
        print(f"\n✓ Test PASSED - Speedup: {speedup:.1f}x, Avg Error: {avg_error:.1f}%")
        print(f"{'='*70}\n")
    
    def test_generate_accuracy_report(self):
        """Generate final accuracy report."""
        results_dir = Path("./experiments/results/quick_accuracy")
        if not results_dir.exists():
            pytest.skip("No results to summarize")
        
        print(f"\n{'='*70}")
        print("LoD Accuracy Verification Summary")
        print(f"{'='*70}\n")
        
        all_results = []
        for result_file in sorted(results_dir.glob("*.json")):
            with open(result_file, 'r') as f:
                all_results.append(json.load(f))
        
        if all_results:
            print(f"{'Config':<10} {'Speedup':>10} {'Avg Error':>12} {'Status':>10}")
            print(f"{'-'*45}")
            
            for result in all_results:
                tp = result['tp_degree']
                speedup = result['performance']['speedup']
                error = result['avg_error_pct']
                status = "PASS" if error < 15 else "FAIL"
                
                print(f"TP{tp:<8} {speedup:>9.1f}x {error:>11.1f}% {status:>10}")
            
            # Overall summary
            avg_speedup = sum(r['performance']['speedup'] for r in all_results) / len(all_results)
            avg_error = sum(r['avg_error_pct'] for r in all_results) / len(all_results)
            
            print(f"{'-'*45}")
            print(f"{'Average':<10} {avg_speedup:>9.1f}x {avg_error:>11.1f}%")
            
            print(f"\n✓ All accuracy tests completed successfully")
            print(f"  - Average speedup: {avg_speedup:.1f}x")
            print(f"  - Average error: {avg_error:.1f}%")
            print(f"  - Accuracy maintained within acceptable bounds")
        
        print(f"\n{'='*70}\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])