"""
Comprehensive accuracy verification tests for LoD abstractions.

This test suite ensures that medium LoD maintains acceptable accuracy
compared to high LoD across various metrics and configurations.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List
import pytest
import numpy as np
from llmperforacle.orchestration import ExperimentOrchestrator


class TestLoDAccuracyVerification:
    """Comprehensive accuracy tests for LoD abstractions."""
    
    def create_test_config(self, config_name: str, lod: str, duration: int = 60) -> Dict[str, Any]:
        """Create test configurations for different scenarios."""
        
        base_config = {
            "simulation": {
                "max_simulation_time": duration,
                "random_seed": 42,
                "lod": lod
            },
            "model_characteristics_db_path": "./configs/model_params.json",
            "metrics_config": {
                "output_summary_json_path": f"./experiments/results/accuracy_{config_name}_{lod}.json",
                "output_requests_csv_path": f"./experiments/results/accuracy_{config_name}_{lod}.csv",
                "compute_token_stats": True,
                "compute_percentiles": [50, 90, 95, 99],
                "warm_up_duration_s": 5
            }
        }
        
        if config_name == "single_gpu":
            # Single GPU configuration
            base_config.update({
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
                        "link_id": "client_to_server",
                        "source_id": "client_node_0",
                        "dest_id": "gpu0",
                        "bandwidth_bps": 10_000_000_000,
                        "latency_s": 0.0001,
                        "bidirectional": True
                    }]
                },
                "workload": self._create_workload(duration, rate=2.0),
                "frameworks_to_test": [{
                    "name": "vllm_single",
                    "type": "VLLM",
                    "is_target_for_workload": True,
                    "config": {
                        "model_profile_id": "Llama2-7B",
                        "gpu_id": "gpu0",
                        "block_size": 16,
                        "max_num_seqs": 256,
                        "enable_prefix_caching": False,
                        "enable_cross_request_caching": False,
                        "enable_chunked_prefill": True,
                        "prefill_chunk_size": 2048
                    }
                }]
            })
            
        elif config_name == "tp4":
            # TP4 configuration
            gpus = [f"gpu{i}" for i in range(4)]
            base_config.update({
                "hardware_profile": self._create_multi_gpu_hardware(4),
                "workload": self._create_workload(duration, rate=8.0),
                "frameworks_to_test": [{
                    "name": "vllm_tp4",
                    "type": "ParallelVLLM",
                    "is_target_for_workload": True,
                    "config": {
                        "model_profile_id": "Llama2-13B",
                        "block_size": 16,
                        "max_num_seqs": 256,
                        "enable_prefix_caching": True,
                        "enable_cross_request_caching": False,
                        "enable_chunked_prefill": True,
                        "prefill_chunk_size": 4096,
                        "parallelism": {
                            "strategy": "TP",
                            "tp_degree": 4,
                            "gpu_ids": gpus
                        }
                    }
                }]
            })
            
        elif config_name == "tp8":
            # TP8 configuration with heavier workload
            gpus = [f"gpu{i}" for i in range(8)]
            base_config.update({
                "hardware_profile": self._create_multi_gpu_hardware(8),
                "workload": self._create_workload(duration, rate=20.0),
                "frameworks_to_test": [{
                    "name": "vllm_tp8",
                    "type": "ParallelVLLM",
                    "is_target_for_workload": True,
                    "config": {
                        "model_profile_id": "Llama3-70B",
                        "block_size": 16,
                        "max_num_seqs": 512,
                        "enable_prefix_caching": True,
                        "enable_cross_request_caching": True,
                        "enable_chunked_prefill": True,
                        "prefill_chunk_size": 8192,
                        "max_num_batched_tokens": 16384,
                        "parallelism": {
                            "strategy": "TP",
                            "tp_degree": 8,
                            "gpu_ids": gpus
                        }
                    }
                }]
            })
            
        return base_config
    
    def _create_workload(self, duration: int, rate: float) -> Dict[str, Any]:
        """Create workload configuration."""
        return {
            "total_duration": duration,
            "bytes_per_token_estimate_for_network": 2,
            "random_seed": 123,
            "max_turns_per_session": 2,
            "client_profiles": [{
                "profile_name": "test_load",
                "weight": 1.0,
                "inter_arrival_time_dist_config": {
                    "type": "Exponential",
                    "rate": rate
                },
                "prompt_tokens_dist_config": {
                    "type": "LogNormal",
                    "mean": 5.5,  # ~245 tokens average
                    "sigma": 0.6,
                    "is_int": True
                },
                "max_output_tokens_dist_config": {
                    "type": "LogNormal",
                    "mean": 5.0,  # ~148 tokens average
                    "sigma": 0.5,
                    "is_int": True
                },
                "conversational_probability": 0.3,
                "streaming_response_probability": 0.8
            }]
        }
    
    def _create_multi_gpu_hardware(self, num_gpus: int) -> Dict[str, Any]:
        """Create multi-GPU hardware configuration."""
        return {
            "compute_devices": [
                {
                    "device_id": f"gpu{i}",
                    "device_type": "GPU",
                    "peak_tflops": {"fp16": 312, "int8": 624},
                    "memory_capacity_bytes": 80_000_000_000,
                    "memory_gbps": 2039,
                    "processing_units": 108
                }
                for i in range(num_gpus)
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
                    "bandwidth_bps": 600_000_000_000,  # NVLink
                    "latency_s": 0.000001,
                    "bidirectional": True
                }
                for i in range(num_gpus)
                for j in range(num_gpus)
                if i != j
            ]
        }
    
    def run_simulation(self, config: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Run simulation and return wall time and report."""
        start_time = time.time()
        orchestrator = ExperimentOrchestrator(config)
        report = orchestrator.run()
        wall_time = time.time() - start_time
        return wall_time, report
    
    def extract_metrics(self, report: Dict[str, Any]) -> Dict[str, float]:
        """Extract key metrics from report."""
        latency = report.get('latency', {})
        requests = report.get('requests', {})
        throughput = report.get('throughput', {})
        token_stats = report.get('token_stats', {})
        
        return {
            # Request metrics
            'total_requests': float(requests.get('total', 0)),
            'successful_requests': float(requests.get('successful', 0)),
            'failed_requests': float(requests.get('failed', 0)),
            'success_rate': float(requests.get('success_rate', 0)),
            
            # Latency metrics (mean)
            'avg_ttft_ms': float(latency.get('time_to_first_token_ms', {}).get('mean', 0)),
            'avg_tpot_ms': float(latency.get('time_per_output_token_ms', {}).get('mean', 0)),
            'avg_e2e_ms': float(latency.get('end_to_end_latency_ms', {}).get('mean', 0)),
            
            # Latency metrics (percentiles)
            'p50_ttft_ms': float(latency.get('time_to_first_token_ms', {}).get('p50', 0)),
            'p90_ttft_ms': float(latency.get('time_to_first_token_ms', {}).get('p90', 0)),
            'p95_ttft_ms': float(latency.get('time_to_first_token_ms', {}).get('p95', 0)),
            'p99_ttft_ms': float(latency.get('time_to_first_token_ms', {}).get('p99', 0)),
            
            'p50_e2e_ms': float(latency.get('end_to_end_latency_ms', {}).get('p50', 0)),
            'p90_e2e_ms': float(latency.get('end_to_end_latency_ms', {}).get('p90', 0)),
            'p95_e2e_ms': float(latency.get('end_to_end_latency_ms', {}).get('p95', 0)),
            'p99_e2e_ms': float(latency.get('end_to_end_latency_ms', {}).get('p99', 0)),
            
            # Throughput metrics
            'request_throughput': float(throughput.get('request_throughput_per_s', 0)),
            'token_throughput': float(throughput.get('tokens_throughput_per_s', 0)),
            
            # Token statistics
            'avg_prompt_tokens': float(token_stats.get('prompt_tokens', {}).get('mean', 0)),
            'avg_output_tokens': float(token_stats.get('output_tokens', {}).get('mean', 0)),
        }
    
    def calculate_accuracy_metrics(
        self, 
        high_metrics: Dict[str, float], 
        medium_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate accuracy metrics between high and medium LoD."""
        accuracy_metrics = {}
        
        for key in high_metrics:
            high_val = high_metrics[key]
            medium_val = medium_metrics[key]
            
            if high_val > 0:
                # Calculate relative error
                rel_error = abs(medium_val - high_val) / high_val * 100
                accuracy_metrics[f'{key}_error_pct'] = rel_error
                
                # Calculate ratio
                accuracy_metrics[f'{key}_ratio'] = medium_val / high_val
            else:
                accuracy_metrics[f'{key}_error_pct'] = 0.0 if medium_val == 0 else float('inf')
                accuracy_metrics[f'{key}_ratio'] = 1.0 if medium_val == 0 else float('inf')
        
        # Calculate aggregate metrics
        error_values = [v for k, v in accuracy_metrics.items() if k.endswith('_error_pct') and v != float('inf')]
        if error_values:
            accuracy_metrics['avg_error_pct'] = np.mean(error_values)
            accuracy_metrics['max_error_pct'] = np.max(error_values)
            accuracy_metrics['median_error_pct'] = np.median(error_values)
        
        return accuracy_metrics
    
    @pytest.mark.parametrize("config_name", ["single_gpu", "tp4", "tp8"])
    def test_lod_accuracy(self, config_name):
        """Test accuracy of medium LoD compared to high LoD."""
        print(f"\n{'='*80}")
        print(f"LoD Accuracy Test: {config_name}")
        print(f"{'='*80}")
        
        # Run simulations
        high_config = self.create_test_config(config_name, "high")
        medium_config = self.create_test_config(config_name, "medium")
        
        print(f"\nRunning HIGH LoD simulation for {config_name}...")
        high_time, high_report = self.run_simulation(high_config)
        
        print(f"Running MEDIUM LoD simulation for {config_name}...")
        medium_time, medium_report = self.run_simulation(medium_config)
        
        # Extract metrics
        high_metrics = self.extract_metrics(high_report)
        medium_metrics = self.extract_metrics(medium_report)
        
        # Calculate accuracy
        accuracy = self.calculate_accuracy_metrics(high_metrics, medium_metrics)
        
        # Calculate speedup
        speedup = high_time / medium_time
        
        # Print results
        print(f"\nPerformance:")
        print(f"  High LoD time:   {high_time:.2f}s")
        print(f"  Medium LoD time: {medium_time:.2f}s")
        print(f"  Speedup:         {speedup:.2f}x")
        
        print(f"\nKey Metrics Comparison:")
        print(f"  {'Metric':<25} {'High LoD':>12} {'Medium LoD':>12} {'Error %':>10}")
        print(f"  {'-'*60}")
        
        key_metrics = [
            'successful_requests', 'success_rate', 
            'avg_ttft_ms', 'avg_tpot_ms', 'avg_e2e_ms',
            'p99_ttft_ms', 'p99_e2e_ms',
            'request_throughput', 'token_throughput'
        ]
        
        for metric in key_metrics:
            high_val = high_metrics[metric]
            medium_val = medium_metrics[metric]
            error = accuracy.get(f'{metric}_error_pct', 0)
            print(f"  {metric:<25} {high_val:>12.2f} {medium_val:>12.2f} {error:>9.1f}%")
        
        print(f"\nAccuracy Summary:")
        print(f"  Average error: {accuracy.get('avg_error_pct', 0):.2f}%")
        print(f"  Maximum error: {accuracy.get('max_error_pct', 0):.2f}%")
        print(f"  Median error:  {accuracy.get('median_error_pct', 0):.2f}%")
        
        # Save detailed results
        results = {
            "config_name": config_name,
            "performance": {
                "high_lod_time": high_time,
                "medium_lod_time": medium_time,
                "speedup": speedup
            },
            "metrics": {
                "high_lod": high_metrics,
                "medium_lod": medium_metrics
            },
            "accuracy": accuracy
        }
        
        results_dir = Path("./experiments/results/accuracy_tests")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / f"{config_name}_accuracy.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Assertions
        assert speedup > 1.0, f"Expected speedup > 1.0, got {speedup:.2f}"
        
        # Check accuracy thresholds
        max_acceptable_error = {
            'single_gpu': 5.0,  # Tighter for single GPU
            'tp4': 10.0,        # Medium for TP4
            'tp8': 15.0         # More lenient for TP8 due to complexity
        }
        
        threshold = max_acceptable_error.get(config_name, 10.0)
        assert accuracy.get('avg_error_pct', 0) < threshold, \
            f"Average error {accuracy.get('avg_error_pct', 0):.2f}% exceeds threshold {threshold}%"
        
        # Check critical metrics individually
        critical_metrics = ['successful_requests', 'avg_ttft_ms', 'avg_e2e_ms', 'request_throughput']
        for metric in critical_metrics:
            error = accuracy.get(f'{metric}_error_pct', 0)
            assert error < threshold * 1.5, \
                f"{metric} error {error:.2f}% exceeds critical threshold {threshold * 1.5}%"
        
        print(f"\n✓ Test PASSED for {config_name}")
        print(f"{'='*80}\n")
    
    def test_accuracy_summary(self):
        """Generate summary report of all accuracy tests."""
        results_dir = Path("./experiments/results/accuracy_tests")
        if not results_dir.exists():
            pytest.skip("No accuracy test results found")
        
        print(f"\n{'='*80}")
        print("LoD Accuracy Test Summary")
        print(f"{'='*80}\n")
        
        summary = {
            "configs": [],
            "overall_stats": {
                "avg_speedup": 0,
                "avg_error": 0,
                "max_error": 0
            }
        }
        
        # Load all results
        speedups = []
        errors = []
        
        for result_file in sorted(results_dir.glob("*_accuracy.json")):
            with open(result_file, 'r') as f:
                result = json.load(f)
            
            summary["configs"].append(result)
            speedups.append(result["performance"]["speedup"])
            errors.append(result["accuracy"].get("avg_error_pct", 0))
        
        if speedups:
            summary["overall_stats"]["avg_speedup"] = np.mean(speedups)
            summary["overall_stats"]["avg_error"] = np.mean(errors)
            summary["overall_stats"]["max_error"] = np.max(errors)
        
        # Print summary table
        print(f"{'Configuration':<15} {'Speedup':>10} {'Avg Error':>12} {'Max Error':>12}")
        print(f"{'-'*50}")
        
        for config in summary["configs"]:
            print(f"{config['config_name']:<15} "
                  f"{config['performance']['speedup']:>9.1f}x "
                  f"{config['accuracy'].get('avg_error_pct', 0):>11.1f}% "
                  f"{config['accuracy'].get('max_error_pct', 0):>11.1f}%")
        
        print(f"{'-'*50}")
        print(f"{'Overall':<15} "
              f"{summary['overall_stats']['avg_speedup']:>9.1f}x "
              f"{summary['overall_stats']['avg_error']:>11.1f}% "
              f"{summary['overall_stats']['max_error']:>11.1f}%")
        
        # Save summary
        with open(results_dir / "accuracy_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ All accuracy tests completed successfully")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])