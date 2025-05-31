"""Integration tests for prefix caching in full simulations."""

import json
import tempfile
from pathlib import Path

import pytest

from llmperforacle.orchestration import ExperimentOrchestrator


class TestPrefixCachingIntegration:
    """Integration tests for prefix caching with conversational workloads."""
    
    @pytest.fixture
    def base_config(self):
        """Base configuration for prefix caching tests."""
        return {
            "simulation": {
                "max_simulation_time": 20,
                "random_seed": 42
            },
            "model_characteristics_db_path": "configs/model_params.json",
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
                        "dest_id": "framework_entry_0",
                        "bandwidth_bps": 10_000_000_000,
                        "latency_s": 0.0001,
                        "bidirectional": True
                    }
                ]
            },
            "frameworks_to_test": [
                {
                    "name": "vllm_with_cache",
                    "type": "VLLM",
                    "is_target_for_workload": True,
                    "config": {
                        "model_profile_id": "Llama2-7B",
                        "gpu_id": "gpu0",
                        "block_size": 16,
                        "max_num_seqs": 32,
                        "max_num_batched_tokens": 2048,
                        "scheduler_iteration_delay_s": 0.0001,
                        "enable_prefix_caching": True
                    }
                }
            ],
            "metrics_config": {
                "percentiles_to_calculate": [0.5, 0.9],
                "warm_up_duration_s": 2
            }
        }
    
    def test_conversational_workload_with_caching(self, base_config, tmp_path):
        """Test that conversational workloads benefit from prefix caching."""
        config = base_config.copy()
        
        # Create conversational workload
        config["workload"] = {
            "total_duration": 20,
            "bytes_per_token_estimate_for_network": 2,
            "client_profiles": [
                {
                    "profile_name": "conversational",
                    "weight": 1.0,
                    "inter_arrival_time_dist_config": {
                        "type": "Exponential",
                        "rate": 2.0  # 2 req/s
                    },
                    "prompt_tokens_dist_config": {
                        "type": "Constant",
                        "value": 500,
                        "is_int": True
                    },
                    "max_output_tokens_dist_config": {
                        "type": "Constant",
                        "value": 100,
                        "is_int": True
                    },
                    "conversational_probability": 0.8,  # 80% chance of follow-up
                    "follow_up_inter_arrival_time_dist_config": {
                        "type": "Constant",
                        "value": 0.5  # Quick follow-up
                    },
                    "streaming_response_probability": 0.0
                }
            ]
        }
        
        config["metrics_config"]["output_summary_json_path"] = str(tmp_path / "conversational_cache.json")
        config["metrics_config"]["output_requests_csv_path"] = str(tmp_path / "conversational_cache.csv")
        
        # Run simulation
        orchestrator = ExperimentOrchestrator(config)
        orchestrator.run()
        
        # Load results
        with open(tmp_path / "conversational_cache.json") as f:
            results = json.load(f)
        
        # Verify prefix caching worked
        assert "prefix_caching" in results
        cache_stats = results["prefix_caching"]
        
        # Should have reasonable hit rate for conversational workload
        assert cache_stats["overall_hit_rate"] > 0.3, f"Hit rate too low: {cache_stats['overall_hit_rate']:.2%}"
        assert cache_stats["conversational_hit_rate"] > 0.5, f"Conversational hit rate too low: {cache_stats['conversational_hit_rate']:.2%}"
        
        # Should save significant tokens
        assert cache_stats["prefill_reduction_ratio"] > 0.2, f"Prefill reduction too low: {cache_stats['prefill_reduction_ratio']:.2%}"
        
        print(f"\nConversational Workload Results:")
        print(f"Overall Hit Rate: {cache_stats['overall_hit_rate']:.1%}")
        print(f"Conversational Hit Rate: {cache_stats['conversational_hit_rate']:.1%}")
        print(f"Prefill Reduction: {cache_stats['prefill_reduction_ratio']:.1%}")
        print(f"Tokens Saved: {cache_stats['total_tokens_saved']:,}")
    
    def test_prefix_caching_performance_benefit(self, base_config, tmp_path):
        """Test that prefix caching improves TTFT for conversational workloads."""
        # Run with caching enabled
        config_with_cache = base_config.copy()
        config_with_cache["workload"] = {
            "total_duration": 20,
            "bytes_per_token_estimate_for_network": 2,
            "client_profiles": [
                {
                    "profile_name": "conversational",
                    "weight": 1.0,
                    "inter_arrival_time_dist_config": {
                        "type": "Exponential",
                        "rate": 1.0  # 1 req/s
                    },
                    "prompt_tokens_dist_config": {
                        "type": "Linear",
                        "start": 200,
                        "end": 800,  # Growing context over conversation
                        "is_int": True
                    },
                    "max_output_tokens_dist_config": {
                        "type": "Constant",
                        "value": 50,
                        "is_int": True
                    },
                    "conversational_probability": 0.9,
                    "follow_up_inter_arrival_time_dist_config": {
                        "type": "Constant",
                        "value": 0.2
                    },
                    "streaming_response_probability": 0.0
                }
            ]
        }
        
        config_with_cache["metrics_config"]["output_summary_json_path"] = str(tmp_path / "with_cache.json")
        config_with_cache["metrics_config"]["output_requests_csv_path"] = str(tmp_path / "with_cache.csv")
        
        orchestrator = ExperimentOrchestrator(config_with_cache)
        orchestrator.run()
        
        with open(tmp_path / "with_cache.json") as f:
            results_with_cache = json.load(f)
        
        # Run with caching disabled
        config_no_cache = config_with_cache.copy()
        config_no_cache["frameworks_to_test"][0]["config"]["enable_prefix_caching"] = False
        config_no_cache["metrics_config"]["output_summary_json_path"] = str(tmp_path / "no_cache.json")
        config_no_cache["metrics_config"]["output_requests_csv_path"] = str(tmp_path / "no_cache.csv")
        
        orchestrator = ExperimentOrchestrator(config_no_cache)
        orchestrator.run()
        
        with open(tmp_path / "no_cache.json") as f:
            results_no_cache = json.load(f)
        
        # Compare TTFT
        ttft_with_cache = results_with_cache["latency"]["time_to_first_token_ms"]["p50"]
        ttft_no_cache = results_no_cache["latency"]["time_to_first_token_ms"]["p50"]
        
        improvement = (ttft_no_cache - ttft_with_cache) / ttft_no_cache
        
        print(f"\nPrefix Caching Performance Impact:")
        print(f"TTFT without cache: {ttft_no_cache:.1f}ms")
        print(f"TTFT with cache: {ttft_with_cache:.1f}ms")
        print(f"Improvement: {improvement:.1%}")
        
        # Prefix caching should improve TTFT
        assert ttft_with_cache < ttft_no_cache, "Prefix caching should reduce TTFT"
        assert improvement > 0.1, f"Expected >10% TTFT improvement, got {improvement:.1%}"
    
    def test_non_conversational_workload(self, base_config, tmp_path):
        """Test that non-conversational workloads show minimal cache hits."""
        config = base_config.copy()
        
        # Create non-conversational workload
        config["workload"] = {
            "total_duration": 20,
            "bytes_per_token_estimate_for_network": 2,
            "client_profiles": [
                {
                    "profile_name": "independent",
                    "weight": 1.0,
                    "inter_arrival_time_dist_config": {
                        "type": "Exponential",
                        "rate": 2.0
                    },
                    "prompt_tokens_dist_config": {
                        "type": "Uniform",
                        "low": 100,
                        "high": 500,
                        "is_int": True
                    },
                    "max_output_tokens_dist_config": {
                        "type": "Uniform",
                        "low": 50,
                        "high": 150,
                        "is_int": True
                    },
                    "conversational_probability": 0.0,  # No conversations
                    "streaming_response_probability": 0.0
                }
            ]
        }
        
        config["metrics_config"]["output_summary_json_path"] = str(tmp_path / "non_conversational.json")
        config["metrics_config"]["output_requests_csv_path"] = str(tmp_path / "non_conversational.csv")
        
        # Run simulation
        orchestrator = ExperimentOrchestrator(config)
        orchestrator.run()
        
        # Load results
        with open(tmp_path / "non_conversational.json") as f:
            results = json.load(f)
        
        # Verify minimal caching
        if "prefix_caching" in results:
            cache_stats = results["prefix_caching"]
            # Should have very low hit rate
            assert cache_stats["overall_hit_rate"] < 0.1, f"Hit rate too high for non-conversational: {cache_stats['overall_hit_rate']:.2%}"
            assert cache_stats["prefill_reduction_ratio"] < 0.1, f"Too much prefill reduction: {cache_stats['prefill_reduction_ratio']:.2%}"
            
            print(f"\nNon-Conversational Workload Results:")
            print(f"Overall Hit Rate: {cache_stats['overall_hit_rate']:.1%}")
            print(f"Prefill Reduction: {cache_stats['prefill_reduction_ratio']:.1%}")
        else:
            print("\nNo prefix caching events recorded (expected for non-conversational workload)")
    
    def test_mixed_workload(self, base_config, tmp_path):
        """Test prefix caching with mixed conversational and non-conversational requests."""
        config = base_config.copy()
        
        # Create mixed workload
        config["workload"] = {
            "total_duration": 20,
            "bytes_per_token_estimate_for_network": 2,
            "client_profiles": [
                {
                    "profile_name": "conversational_users",
                    "weight": 0.6,  # 60% of traffic
                    "inter_arrival_time_dist_config": {
                        "type": "Exponential",
                        "rate": 1.5
                    },
                    "prompt_tokens_dist_config": {
                        "type": "Constant",
                        "value": 400,
                        "is_int": True
                    },
                    "max_output_tokens_dist_config": {
                        "type": "Constant",
                        "value": 100,
                        "is_int": True
                    },
                    "conversational_probability": 0.8,
                    "follow_up_inter_arrival_time_dist_config": {
                        "type": "Exponential",
                        "rate": 2.0
                    },
                    "streaming_response_probability": 0.0
                },
                {
                    "profile_name": "one_shot_users",
                    "weight": 0.4,  # 40% of traffic
                    "inter_arrival_time_dist_config": {
                        "type": "Exponential",
                        "rate": 1.0
                    },
                    "prompt_tokens_dist_config": {
                        "type": "Uniform",
                        "low": 50,
                        "high": 300,
                        "is_int": True
                    },
                    "max_output_tokens_dist_config": {
                        "type": "Uniform",
                        "low": 20,
                        "high": 100,
                        "is_int": True
                    },
                    "conversational_probability": 0.0,
                    "streaming_response_probability": 0.0
                }
            ]
        }
        
        config["metrics_config"]["output_summary_json_path"] = str(tmp_path / "mixed_workload.json")
        config["metrics_config"]["output_requests_csv_path"] = str(tmp_path / "mixed_workload.csv")
        
        # Run simulation
        orchestrator = ExperimentOrchestrator(config)
        orchestrator.run()
        
        # Load results
        with open(tmp_path / "mixed_workload.json") as f:
            results = json.load(f)
        
        # Verify caching stats reflect mixed workload
        assert "prefix_caching" in results
        cache_stats = results["prefix_caching"]
        
        # Should have moderate hit rate
        assert 0.2 < cache_stats["overall_hit_rate"] < 0.6, f"Unexpected hit rate: {cache_stats['overall_hit_rate']:.2%}"
        
        # Check event distribution
        events = cache_stats["event_counts"]
        total_events = sum(events.values())
        conversational_hits = events.get("CONVERSATIONAL_HIT", 0)
        full_misses = events.get("MISS_FULL", 0)
        
        print(f"\nMixed Workload Results:")
        print(f"Overall Hit Rate: {cache_stats['overall_hit_rate']:.1%}")
        print(f"Conversational Hits: {conversational_hits}/{total_events} ({conversational_hits/total_events*100:.1%})")
        print(f"Full Misses: {full_misses}/{total_events} ({full_misses/total_events*100:.1%})")
        print(f"Prefill Reduction: {cache_stats['prefill_reduction_ratio']:.1%}")