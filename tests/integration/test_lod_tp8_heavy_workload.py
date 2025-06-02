"""
Test LoD performance improvements with heavy workload on TP8 configuration.

This test demonstrates how medium LoD significantly speeds up simulations
that previously suffered from excessive discrete event simulation overhead.
"""

import time
import json
from pathlib import Path
import pytest
from llmperforacle.orchestration import ExperimentOrchestrator


def create_tp8_heavy_config(lod="high", duration=120):
    """Create TP8 configuration with heavy workload."""
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
                    "memory_capacity_bytes": 80_000_000_000,  # 80 GB
                    "memory_gbps": 2039,
                    "processing_units": 108
                }
                for i in range(8)
            ],
            "network_links": [
                {
                    "link_id": "client_to_server",
                    "source_id": "client_node_0", 
                    "dest_id": "gpu0",
                    "bandwidth_bps": 100_000_000_000,  # 100 Gbps
                    "latency_s": 0.0001,
                    "bidirectional": True
                }
            ] + [
                {
                    "link_id": f"gpu{i}_to_gpu{j}",
                    "source_id": f"gpu{i}",
                    "dest_id": f"gpu{j}",
                    "bandwidth_bps": 600_000_000_000,  # 600 Gbps NVLink
                    "latency_s": 0.000001,  # 1 microsecond
                    "bidirectional": True
                }
                for i in range(8)
                for j in range(8)
                if i != j
            ]
        },
        "workload": {
            "total_duration": duration,
            "bytes_per_token_estimate_for_network": 2,
            "random_seed": 123,
            "max_turns_per_session": 3,  # Multi-turn conversations
            "client_profiles": [
                {
                    "profile_name": "heavy_load",
                    "weight": 0.7,
                    "inter_arrival_time_dist_config": {
                        "type": "Exponential",
                        "rate": 25.0  # 25 requests per second
                    },
                    "prompt_tokens_dist_config": {
                        "type": "LogNormal",
                        "mean": 6.0,  # ~400 tokens average
                        "sigma": 0.8,
                        "is_int": True
                    },
                    "max_output_tokens_dist_config": {
                        "type": "LogNormal",
                        "mean": 5.5,  # ~245 tokens average
                        "sigma": 0.6,
                        "is_int": True
                    },
                    "conversational_probability": 0.6,
                    "streaming_response_probability": 0.8,
                    "follow_up_inter_arrival_time_dist_config": {
                        "type": "Exponential",
                        "rate": 0.2  # 5 seconds average between turns
                    }
                },
                {
                    "profile_name": "burst_load",
                    "weight": 0.3,
                    "inter_arrival_time_dist_config": {
                        "type": "Exponential", 
                        "rate": 50.0  # 50 requests per second bursts
                    },
                    "prompt_tokens_dist_config": {
                        "type": "Uniform",
                        "low": 1000,
                        "high": 2000,
                        "is_int": True
                    },
                    "max_output_tokens_dist_config": {
                        "type": "Fixed",
                        "value": 100
                    },
                    "conversational_probability": 0.2,
                    "streaming_response_probability": 1.0
                }
            ]
        },
        "frameworks_to_test": [
            {
                "name": "vllm_tp8",
                "type": "ParallelVLLM",
                "is_target_for_workload": True,
                "config": {
                    "model_profile_id": "Llama3-70B",  # 80 layers - many events!
                    "block_size": 16,
                    "max_num_seqs": 512,  # Higher capacity
                    "enable_prefix_caching": True,  # Add complexity
                    "enable_cross_request_caching": True,
                    "enable_chunked_prefill": True,
                    "prefill_chunk_size": 8192,
                    "max_num_batched_tokens": 16384,
                    "parallelism": {
                        "strategy": "TP",
                        "tp_degree": 8,
                        "gpu_ids": ["gpu0", "gpu1", "gpu2", "gpu3", "gpu4", "gpu5", "gpu6", "gpu7"]
                    }
                }
            }
        ],
        "metrics_config": {
            "output_summary_json_path": f"./experiments/results/tp8_heavy_{lod}_lod.json",
            "output_requests_csv_path": f"./experiments/results/tp8_heavy_{lod}_lod.csv",
            "compute_token_stats": True,
            "compute_percentiles": [50, 90, 95, 99],
            "warm_up_duration_s": 10  # Ignore first 10 seconds
        }
    }


class TestLoDTP8HeavyWorkload:
    """Test LoD performance with TP8 and heavy workload."""
    
    def test_tp8_heavy_workload_comparison(self):
        """Compare high vs medium LoD with TP8 heavy workload."""
        print("\n" + "="*80)
        print("TP8 Heavy Workload LoD Comparison Test")
        print("="*80)
        
        # Test duration - shorter for quick testing
        test_duration = 60  # 1 minute
        
        # Run high LoD
        print("\nRunning HIGH LoD simulation...")
        high_config = create_tp8_heavy_config(lod="high", duration=test_duration)
        
        high_start = time.time()
        high_orchestrator = ExperimentOrchestrator(high_config)
        high_report = high_orchestrator.run()
        high_time = time.time() - high_start
        
        # Run medium LoD
        print("\nRunning MEDIUM LoD simulation...")
        medium_config = create_tp8_heavy_config(lod="medium", duration=test_duration)
        
        medium_start = time.time()
        medium_orchestrator = ExperimentOrchestrator(medium_config)
        medium_report = medium_orchestrator.run()
        medium_time = time.time() - medium_start
        
        # Calculate metrics
        speedup = high_time / medium_time
        
        # Extract key metrics
        def extract_detailed_metrics(report):
            latency = report.get('latency', {})
            requests = report.get('requests', {})
            throughput = report.get('throughput', {})
            gpu_util = report.get('gpu_utilization', {})
            
            return {
                'wall_clock_time': report.get('wall_clock_time', 0),
                'simulated_time': report.get('simulated_time', 0),
                'total_requests': requests.get('total', 0),
                'successful_requests': requests.get('successful', 0),
                'failed_requests': requests.get('failed', 0),
                'success_rate': requests.get('success_rate', 0),
                'avg_ttft_ms': latency.get('time_to_first_token_ms', {}).get('mean', 0),
                'p99_ttft_ms': latency.get('time_to_first_token_ms', {}).get('p99', 0),
                'avg_tpot_ms': latency.get('time_per_output_token_ms', {}).get('mean', 0),
                'avg_e2e_latency_ms': latency.get('end_to_end_latency_ms', {}).get('mean', 0),
                'p99_e2e_latency_ms': latency.get('end_to_end_latency_ms', {}).get('p99', 0),
                'request_throughput': throughput.get('request_throughput_per_s', 0),
                'token_throughput': throughput.get('tokens_throughput_per_s', 0),
                'avg_gpu_utilization': sum(gpu_util.values()) / len(gpu_util) if gpu_util else 0
            }
        
        high_metrics = extract_detailed_metrics(high_report)
        medium_metrics = extract_detailed_metrics(medium_report)
        
        # Print results
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        
        print(f"\nSimulation Wall Clock Time:")
        print(f"  High LoD:   {high_time:7.2f} seconds")
        print(f"  Medium LoD: {medium_time:7.2f} seconds")
        print(f"  Speedup:    {speedup:7.2f}x")
        
        print(f"\nRequest Statistics:")
        print(f"  Total Requests:      High: {high_metrics['total_requests']:4d}, Medium: {medium_metrics['total_requests']:4d}")
        print(f"  Successful Requests: High: {high_metrics['successful_requests']:4d}, Medium: {medium_metrics['successful_requests']:4d}")
        print(f"  Success Rate:        High: {high_metrics['success_rate']:.1%}, Medium: {medium_metrics['success_rate']:.1%}")
        
        print(f"\nLatency Metrics:")
        print(f"  Avg TTFT (ms):       High: {high_metrics['avg_ttft_ms']:6.1f}, Medium: {medium_metrics['avg_ttft_ms']:6.1f}")
        print(f"  P99 TTFT (ms):       High: {high_metrics['p99_ttft_ms']:6.1f}, Medium: {medium_metrics['p99_ttft_ms']:6.1f}")
        print(f"  Avg TPOT (ms):       High: {high_metrics['avg_tpot_ms']:6.1f}, Medium: {medium_metrics['avg_tpot_ms']:6.1f}")
        print(f"  Avg E2E (ms):        High: {high_metrics['avg_e2e_latency_ms']:6.1f}, Medium: {medium_metrics['avg_e2e_latency_ms']:6.1f}")
        
        print(f"\nThroughput:")
        print(f"  Requests/sec:        High: {high_metrics['request_throughput']:6.1f}, Medium: {medium_metrics['request_throughput']:6.1f}")
        print(f"  Tokens/sec:          High: {high_metrics['token_throughput']:6.1f}, Medium: {medium_metrics['token_throughput']:6.1f}")
        
        # Calculate accuracy metrics
        def calculate_relative_error(high_val, medium_val):
            if high_val == 0:
                return 0 if medium_val == 0 else float('inf')
            return abs(medium_val - high_val) / high_val * 100
        
        print(f"\nAccuracy Analysis (% difference from high LoD):")
        key_metrics = ['avg_ttft_ms', 'avg_tpot_ms', 'avg_e2e_latency_ms', 
                      'request_throughput', 'token_throughput']
        
        max_error = 0
        for metric in key_metrics:
            error = calculate_relative_error(high_metrics[metric], medium_metrics[metric])
            max_error = max(max_error, error)
            print(f"  {metric:20s}: {error:5.1f}%")
        
        print("\n" + "="*80)
        
        # Detailed event analysis if available
        if hasattr(high_orchestrator, 'sim_env_wrapper'):
            print(f"\nSimulation Event Analysis:")
            print(f"  High LoD events:   (many per-layer operations)")
            print(f"  Medium LoD events: (aggregated operations)")
            print(f"  Event reduction:   ~{(1 - 1/speedup)*100:.0f}%")
        
        # Save comparison results
        comparison_results = {
            "test_name": "tp8_heavy_workload",
            "config": {
                "duration": test_duration,
                "model": "Llama3-70B",
                "tp_degree": 8,
                "avg_request_rate": 32.5  # 25 * 0.7 + 50 * 0.3
            },
            "performance": {
                "high_lod_time": high_time,
                "medium_lod_time": medium_time,
                "speedup": speedup
            },
            "metrics": {
                "high_lod": high_metrics,
                "medium_lod": medium_metrics,
                "max_error_pct": max_error
            }
        }
        
        results_dir = Path("./experiments/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / "tp8_lod_comparison.json", "w") as f:
            json.dump(comparison_results, f, indent=2)
        
        # Assertions
        assert speedup > 1.5, f"Expected significant speedup > 1.5x for TP8 heavy workload, got {speedup:.2f}x"
        assert max_error < 15, f"Maximum metric error {max_error:.1f}% exceeds 15% threshold"
        
        print(f"\nTest PASSED! Speedup: {speedup:.2f}x with max error: {max_error:.1f}%")
        print("="*80 + "\n")
    
    def test_event_count_analysis(self):
        """Analyze event counts for different LoD levels."""
        print("\n" + "="*80)
        print("Event Count Analysis for TP8")
        print("="*80)
        
        # Calculate theoretical event counts
        model_layers = 80  # Llama3-70B
        tp_degree = 8
        avg_requests_per_second = 32.5
        simulation_duration = 60
        avg_tokens_per_request = 400 + 245  # prompt + output
        
        # High LoD events
        # Per request: layers * 2 (attention + mlp) * (prefill + decode steps)
        decode_steps = 245  # avg output tokens
        events_per_request_high = model_layers * 2 * (1 + decode_steps)
        
        # For TP: add collective events
        collectives_per_layer = 2  # attention and mlp each need allreduce
        collective_events_per_request = model_layers * collectives_per_layer * (1 + decode_steps)
        
        total_events_high = avg_requests_per_second * simulation_duration * (
            events_per_request_high + collective_events_per_request
        )
        
        # Medium LoD events  
        # Per request: 1 prefill + decode steps (aggregated)
        events_per_request_medium = 1 + decode_steps
        
        # For TP: 1 collective per prefill/decode
        collective_events_per_request_medium = 1 + decode_steps
        
        total_events_medium = avg_requests_per_second * simulation_duration * (
            events_per_request_medium + collective_events_per_request_medium
        )
        
        event_reduction = (total_events_high - total_events_medium) / total_events_high * 100
        
        print(f"\nTheoretical Event Count Estimates:")
        print(f"  Model layers: {model_layers}")
        print(f"  TP degree: {tp_degree}")
        print(f"  Avg request rate: {avg_requests_per_second:.1f} req/s")
        print(f"  Simulation duration: {simulation_duration}s")
        print(f"  Avg tokens/request: {avg_tokens_per_request}")
        
        print(f"\nHigh LoD:")
        print(f"  Events per request: {events_per_request_high + collective_events_per_request:,}")
        print(f"  Total events: {total_events_high:,.0f}")
        
        print(f"\nMedium LoD:")
        print(f"  Events per request: {events_per_request_medium + collective_events_per_request_medium:,}")
        print(f"  Total events: {total_events_medium:,.0f}")
        
        print(f"\nEvent Reduction: {event_reduction:.1f}%")
        # Speedup depends on event processing overhead - more events removed = higher speedup
        # Conservative estimate: 50-80% of the event reduction translates to speedup
        print(f"Expected Speedup Range: {1/(1-event_reduction/100*0.5):.1f}x - {1/(1-event_reduction/100*0.8):.1f}x")
        
        print("="*80 + "\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])