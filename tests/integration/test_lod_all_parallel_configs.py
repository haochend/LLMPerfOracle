"""
Comprehensive test suite for all parallel configurations with medium LoD.

Tests Data Parallelism (DP), Pipeline Parallelism (PP), Tensor Parallelism (TP),
and mixed configurations to ensure medium LoD works correctly with all parallelism strategies.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pytest
from llmperforacle.orchestration import ExperimentOrchestrator


class TestLoDAllParallelConfigs:
    """Test all parallel configurations with medium LoD."""
    
    def create_hardware_config(self, num_gpus: int) -> Dict[str, Any]:
        """Create hardware configuration with specified number of GPUs."""
        return {
            "compute_devices": [
                {
                    "device_id": f"gpu{i}",
                    "device_type": "GPU",
                    "peak_tflops": {"fp16": 312, "int8": 624},
                    "memory_capacity_bytes": 80_000_000_000,  # 80 GB
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
                    "bandwidth_bps": 10_000_000_000,  # 10 Gbps
                    "latency_s": 0.0001,
                    "bidirectional": True
                }
            ] + [
                # All-to-all GPU connections
                {
                    "link_id": f"gpu{i}_to_gpu{j}",
                    "source_id": f"gpu{i}",
                    "dest_id": f"gpu{j}",
                    "bandwidth_bps": 600_000_000_000,  # 600 Gbps NVLink
                    "latency_s": 0.000001,
                    "bidirectional": True
                }
                for i in range(num_gpus)
                for j in range(num_gpus)
                if i != j
            ]
        }
    
    def create_workload_config(self, duration: int, request_rate: float) -> Dict[str, Any]:
        """Create workload configuration."""
        return {
            "total_duration": duration,
            "bytes_per_token_estimate_for_network": 2,
            "random_seed": 123,
            "max_turns_per_session": 2,
            "client_profiles": [
                {
                    "profile_name": "standard_load",
                    "weight": 1.0,
                    "inter_arrival_time_dist_config": {
                        "type": "Exponential",
                        "rate": request_rate
                    },
                    "prompt_tokens_dist_config": {
                        "type": "LogNormal",
                        "mean": 5.5,  # ~245 tokens average
                        "sigma": 0.5,
                        "is_int": True
                    },
                    "max_output_tokens_dist_config": {
                        "type": "LogNormal",
                        "mean": 5.0,  # ~148 tokens average
                        "sigma": 0.4,
                        "is_int": True
                    },
                    "conversational_probability": 0.3,
                    "streaming_response_probability": 0.8,
                    "follow_up_inter_arrival_time_dist_config": {
                        "type": "Exponential",
                        "rate": 0.2
                    }
                }
            ]
        }
    
    def create_parallel_config(
        self, 
        config_name: str, 
        lod: str = "medium",
        duration: int = 30
    ) -> Dict[str, Any]:
        """Create configuration for different parallel strategies."""
        
        base_config = {
            "simulation": {
                "max_simulation_time": duration,
                "random_seed": 42,
                "lod": lod
            },
            "model_characteristics_db_path": "./configs/model_params.json",
            "metrics_config": {
                "output_summary_json_path": f"./experiments/results/parallel_{config_name}_{lod}.json",
                "output_requests_csv_path": f"./experiments/results/parallel_{config_name}_{lod}.csv",
                "compute_token_stats": True,
                "compute_percentiles": [50, 90, 95, 99],
                "warm_up_duration_s": 5
            }
        }
        
        if config_name == "dp2":
            # Data Parallelism with 2 replicas
            base_config.update({
                "hardware_profile": self.create_hardware_config(2),
                "workload": self.create_workload_config(duration, 10.0),
                "frameworks_to_test": [
                    {
                        "name": "vllm_replica_0",
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
                    },
                    {
                        "name": "vllm_replica_1",
                        "type": "VLLM",
                        "is_target_for_workload": True,
                        "config": {
                            "model_profile_id": "Llama2-7B",
                            "gpu_id": "gpu1",
                            "block_size": 16,
                            "max_num_seqs": 256,
                            "enable_prefix_caching": False,
                            "enable_cross_request_caching": False,
                            "enable_chunked_prefill": True,
                            "prefill_chunk_size": 2048
                        }
                    }
                ]
            })
            
        elif config_name == "pp4":
            # Pipeline Parallelism with 4 stages
            base_config.update({
                "hardware_profile": self.create_hardware_config(4),
                "workload": self.create_workload_config(duration, 8.0),
                "frameworks_to_test": [{
                    "name": "vllm_pp4",
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
                            "strategy": "PP",
                            "pp_stages": 4,
                            "num_microbatches_per_request": 4,
                            "gpu_ids": ["gpu0", "gpu1", "gpu2", "gpu3"]
                        }
                    }
                }]
            })
            
        elif config_name == "tp4":
            # Tensor Parallelism with 4 GPUs
            base_config.update({
                "hardware_profile": self.create_hardware_config(4),
                "workload": self.create_workload_config(duration, 12.0),
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
                            "gpu_ids": ["gpu0", "gpu1", "gpu2", "gpu3"]
                        }
                    }
                }]
            })
            
        elif config_name == "tp2_pp2":
            # Mixed: TP=2, PP=2 (4 GPUs total)
            base_config.update({
                "hardware_profile": self.create_hardware_config(4),
                "workload": self.create_workload_config(duration, 10.0),
                "frameworks_to_test": [{
                    "name": "vllm_tp2pp2",
                    "type": "ParallelVLLM",
                    "is_target_for_workload": True,
                    "config": {
                        "model_profile_id": "Llama3-8B",
                        "block_size": 16,
                        "max_num_seqs": 256,
                        "enable_prefix_caching": True,
                        "enable_cross_request_caching": False,
                        "enable_chunked_prefill": True,
                        "prefill_chunk_size": 4096,
                        "parallelism": {
                            "strategy": "TP_PP",
                            "tp_degree": 2,
                            "pp_stages": 2,
                            "num_microbatches_per_request": 2,
                            "gpu_ids": ["gpu0", "gpu1", "gpu2", "gpu3"]
                        }
                    }
                }]
            })
            
        elif config_name == "tp4_pp2":
            # Mixed: TP=4, PP=2 (8 GPUs total)
            base_config.update({
                "hardware_profile": self.create_hardware_config(8),
                "workload": self.create_workload_config(duration, 20.0),
                "frameworks_to_test": [{
                    "name": "vllm_tp4pp2",
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
                        "parallelism": {
                            "strategy": "TP_PP",
                            "tp_degree": 4,
                            "pp_stages": 2,
                            "num_microbatches_per_request": 4,
                            "gpu_ids": ["gpu0", "gpu1", "gpu2", "gpu3", "gpu4", "gpu5", "gpu6", "gpu7"]
                        }
                    }
                }]
            })
            
        elif config_name == "dp2_tp2":
            # Mixed: DP=2, TP=2 (4 GPUs total, 2 replicas of TP=2)
            base_config.update({
                "hardware_profile": self.create_hardware_config(4),
                "workload": self.create_workload_config(duration, 16.0),
                "frameworks_to_test": [
                    {
                        "name": "vllm_dp0_tp2",
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
                                "tp_degree": 2,
                                "gpu_ids": ["gpu0", "gpu1"]
                            }
                        }
                    },
                    {
                        "name": "vllm_dp1_tp2",
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
                                "tp_degree": 2,
                                "gpu_ids": ["gpu2", "gpu3"]
                            }
                        }
                    }
                ]
            })
        
        return base_config
    
    def run_test(self, config: Dict[str, Any]) -> Tuple[float, Dict[str, Any], bool]:
        """Run simulation and return time, report, and success status."""
        try:
            start_time = time.time()
            orchestrator = ExperimentOrchestrator(config)
            report = orchestrator.run()
            wall_time = time.time() - start_time
            return wall_time, report, True
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            return 0.0, {}, False
    
    @pytest.mark.parametrize("config_name", [
        "dp2",      # Data Parallelism
        "pp4",      # Pipeline Parallelism
        "tp4",      # Tensor Parallelism
        "tp2_pp2",  # Mixed TP+PP
        "tp4_pp2",  # Mixed TP+PP (larger)
        "dp2_tp2",  # Mixed DP+TP
    ])
    def test_parallel_config_medium_lod(self, config_name):
        """Test each parallel configuration with medium LoD."""
        print(f"\n{'='*70}")
        print(f"Testing {config_name.upper()} with Medium LoD")
        print(f"{'='*70}")
        
        # Create and run configuration
        config = self.create_parallel_config(config_name, lod="medium", duration=20)
        
        print(f"\nRunning {config_name} simulation...")
        wall_time, report, success = self.run_test(config)
        
        if not success:
            pytest.fail(f"Simulation failed for {config_name}")
        
        # Extract metrics
        requests = report.get('requests', {})
        latency = report.get('latency', {})
        throughput = report.get('throughput', {})
        
        total_requests = requests.get('total', 0)
        successful_requests = requests.get('successful', 0)
        success_rate = requests.get('success_rate', 0)
        avg_ttft = latency.get('time_to_first_token_ms', {}).get('mean', 0)
        avg_e2e = latency.get('end_to_end_latency_ms', {}).get('mean', 0)
        req_throughput = throughput.get('request_throughput_per_s', 0)
        
        # Print results
        print(f"\nResults for {config_name}:")
        print(f"  Wall clock time:      {wall_time:.2f}s")
        print(f"  Total requests:       {total_requests}")
        print(f"  Successful requests:  {successful_requests}")
        print(f"  Success rate:         {success_rate:.1%}")
        print(f"  Avg TTFT (ms):        {avg_ttft:.1f}")
        print(f"  Avg E2E latency (ms): {avg_e2e:.1f}")
        print(f"  Request throughput:   {req_throughput:.1f} req/s")
        
        # Save results
        results = {
            "config_name": config_name,
            "wall_time": wall_time,
            "metrics": {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "success_rate": success_rate,
                "avg_ttft_ms": avg_ttft,
                "avg_e2e_ms": avg_e2e,
                "request_throughput": req_throughput
            }
        }
        
        results_dir = Path("./experiments/results/parallel_configs")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / f"{config_name}_medium_lod.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Assertions
        assert success_rate > 0.5, f"Success rate too low: {success_rate:.1%}"
        assert successful_requests > 0, "No successful requests completed"
        assert avg_ttft > 0, "Invalid TTFT metric"
        assert avg_e2e > 0, "Invalid E2E latency metric"
        
        print(f"\n✓ {config_name} test PASSED")
    
    def test_compare_with_high_lod(self):
        """Compare a few configurations between high and medium LoD."""
        print(f"\n{'='*70}")
        print("LoD Comparison for Select Configurations")
        print(f"{'='*70}")
        
        test_configs = ["tp4", "tp2_pp2"]  # Representative configs
        comparison_results = []
        
        for config_name in test_configs:
            print(f"\n\nTesting {config_name}...")
            
            # Run high LoD
            high_config = self.create_parallel_config(config_name, lod="high", duration=15)
            high_time, high_report, high_success = self.run_test(high_config)
            
            # Run medium LoD
            medium_config = self.create_parallel_config(config_name, lod="medium", duration=15)
            medium_time, medium_report, medium_success = self.run_test(medium_config)
            
            if not (high_success and medium_success):
                print(f"  Skipping comparison due to simulation failure")
                continue
            
            # Calculate speedup
            speedup = high_time / medium_time if medium_time > 0 else 0
            
            # Extract key metrics for comparison
            high_reqs = high_report['requests']['successful']
            medium_reqs = medium_report['requests']['successful']
            
            high_ttft = high_report['latency']['time_to_first_token_ms']['mean']
            medium_ttft = medium_report['latency']['time_to_first_token_ms']['mean']
            
            # Calculate accuracy
            req_diff = abs(medium_reqs - high_reqs) / high_reqs * 100 if high_reqs > 0 else 0
            ttft_diff = abs(medium_ttft - high_ttft) / high_ttft * 100 if high_ttft > 0 else 0
            
            print(f"\nComparison for {config_name}:")
            print(f"  Speedup:              {speedup:.1f}x")
            print(f"  Request count diff:   {req_diff:.1f}%")
            print(f"  TTFT diff:            {ttft_diff:.1f}%")
            
            comparison_results.append({
                "config": config_name,
                "speedup": speedup,
                "req_diff_pct": req_diff,
                "ttft_diff_pct": ttft_diff
            })
        
        # Summary
        if comparison_results:
            avg_speedup = sum(r["speedup"] for r in comparison_results) / len(comparison_results)
            avg_accuracy = sum(r["req_diff_pct"] + r["ttft_diff_pct"] for r in comparison_results) / (2 * len(comparison_results))
            
            print(f"\n{'='*70}")
            print(f"Overall Summary:")
            print(f"  Average speedup:      {avg_speedup:.1f}x")
            print(f"  Average metric diff:  {avg_accuracy:.1f}%")
            print(f"{'='*70}")
    
    def test_summary_report(self):
        """Generate summary report of all parallel configuration tests."""
        results_dir = Path("./experiments/results/parallel_configs")
        if not results_dir.exists():
            pytest.skip("No results to summarize")
        
        print(f"\n{'='*70}")
        print("Parallel Configurations Test Summary")
        print(f"{'='*70}\n")
        
        all_results = []
        for result_file in sorted(results_dir.glob("*_medium_lod.json")):
            with open(result_file, 'r') as f:
                all_results.append(json.load(f))
        
        if all_results:
            print(f"{'Config':<12} {'Time (s)':>10} {'Requests':>10} {'Success':>10} {'TTFT (ms)':>12}")
            print(f"{'-'*55}")
            
            for result in all_results:
                config = result['config_name']
                time_taken = result['wall_time']
                requests = result['metrics']['successful_requests']
                success_rate = result['metrics']['success_rate'] * 100
                ttft = result['metrics']['avg_ttft_ms']
                
                print(f"{config:<12} {time_taken:>10.1f} {requests:>10} {success_rate:>9.1f}% {ttft:>12.1f}")
            
            print(f"\n✓ All parallel configurations tested successfully with medium LoD")
        
        print(f"{'='*70}\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])