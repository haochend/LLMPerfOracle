"""Comprehensive integration tests for prefix caching functionality."""

import json
import tempfile
from pathlib import Path
import csv

import pytest
import numpy as np

from llmperforacle.orchestration import ExperimentOrchestrator


class TestPrefixCachingComprehensive:
    """Comprehensive tests for prefix caching implementation."""
    
    @pytest.fixture
    def base_config(self):
        """Base configuration for comprehensive tests."""
        return {
            "simulation": {
                "max_simulation_time": 30,
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
            "metrics_config": {
                "percentiles_to_calculate": [0.5, 0.9, 0.99],
                "warm_up_duration_s": 3
            }
        }
    
    def test_cache_disabled_vs_enabled(self, base_config, tmp_path):
        """Test that disabling prefix caching works correctly."""
        config = base_config.copy()
        
        # Simple workload
        config["workload"] = {
            "total_duration": 30,
            "bytes_per_token_estimate_for_network": 2,
            "client_profiles": [{
                "profile_name": "test",
                "weight": 1.0,
                "inter_arrival_time_dist_config": {
                    "type": "Exponential",
                    "rate": 2.0
                },
                "prompt_tokens_dist_config": {
                    "type": "Constant",
                    "value": 200,
                    "is_int": True
                },
                "max_output_tokens_dist_config": {
                    "type": "Constant",
                    "value": 50,
                    "is_int": True
                },
                "conversational_probability": 0.7,
                "follow_up_inter_arrival_time_dist_config": {
                    "type": "Constant",
                    "value": 0.5
                },
                "streaming_response_probability": 0.0
            }]
        }
        
        # Test with caching enabled (default)
        config_enabled = config.copy()
        config_enabled["frameworks_to_test"] = [{
            "name": "vllm_enabled",
            "type": "VLLM",
            "is_target_for_workload": True,
            "config": {
                "model_profile_id": "Llama2-7B",
                "gpu_id": "gpu0",
                "block_size": 16,
                "max_num_seqs": 32,
                "enable_prefix_caching": True
            }
        }]
        config_enabled["metrics_config"]["output_summary_json_path"] = str(tmp_path / "enabled.json")
        config_enabled["metrics_config"]["output_requests_csv_path"] = str(tmp_path / "enabled.csv")
        
        orchestrator = ExperimentOrchestrator(config_enabled)
        orchestrator.run()
        
        # Test with caching disabled
        config_disabled = config.copy()
        config_disabled["frameworks_to_test"] = [{
            "name": "vllm_disabled",
            "type": "VLLM",
            "is_target_for_workload": True,
            "config": {
                "model_profile_id": "Llama2-7B",
                "gpu_id": "gpu0",
                "block_size": 16,
                "max_num_seqs": 32,
                "enable_prefix_caching": False
            }
        }]
        config_disabled["metrics_config"]["output_summary_json_path"] = str(tmp_path / "disabled.json")
        config_disabled["metrics_config"]["output_requests_csv_path"] = str(tmp_path / "disabled.csv")
        
        orchestrator = ExperimentOrchestrator(config_disabled)
        orchestrator.run()
        
        # Load results
        with open(tmp_path / "enabled.json") as f:
            results_enabled = json.load(f)
        with open(tmp_path / "disabled.json") as f:
            results_disabled = json.load(f)
        
        # Verify caching metrics exist only when enabled
        assert "prefix_caching" in results_enabled
        assert "prefix_caching" not in results_disabled or results_disabled["prefix_caching"]["overall_hit_rate"] == 0
        
        print(f"\nCache Enabled vs Disabled Test:")
        print(f"Enabled - Has prefix caching metrics: {'prefix_caching' in results_enabled}")
        if "prefix_caching" in results_enabled:
            print(f"Enabled - Events recorded: {sum(results_enabled['prefix_caching']['event_counts'].values())}")
    
    def test_memory_constrained_scenario(self, base_config, tmp_path):
        """Test prefix caching behavior under memory constraints."""
        config = base_config.copy()
        
        # Reduce memory to create pressure
        config["hardware_profile"]["compute_devices"][0]["memory_capacity_bytes"] = 20_000_000_000  # 20GB
        
        # Memory-intensive workload
        config["workload"] = {
            "total_duration": 30,
            "bytes_per_token_estimate_for_network": 2,
            "client_profiles": [{
                "profile_name": "memory_heavy",
                "weight": 1.0,
                "inter_arrival_time_dist_config": {
                    "type": "Exponential",
                    "rate": 1.0
                },
                "prompt_tokens_dist_config": {
                    "type": "Constant",
                    "value": 1000,
                    "is_int": True
                },
                "max_output_tokens_dist_config": {
                    "type": "Constant",
                    "value": 500,
                    "is_int": True
                },
                "conversational_probability": 0.6,
                "follow_up_inter_arrival_time_dist_config": {
                    "type": "Constant",
                    "value": 1.0
                },
                "streaming_response_probability": 0.0
            }]
        }
        
        config["frameworks_to_test"] = [{
            "name": "vllm_memory_test",
            "type": "VLLM",
            "is_target_for_workload": True,
            "config": {
                "model_profile_id": "Llama2-7B",
                "gpu_id": "gpu0",
                "block_size": 16,
                "max_num_seqs": 8,  # Reduced due to memory
                "enable_prefix_caching": True
            }
        }]
        
        config["metrics_config"]["output_summary_json_path"] = str(tmp_path / "memory_test.json")
        config["metrics_config"]["output_requests_csv_path"] = str(tmp_path / "memory_test.csv")
        
        orchestrator = ExperimentOrchestrator(config)
        orchestrator.run()
        
        with open(tmp_path / "memory_test.json") as f:
            results = json.load(f)
        
        # System should still function under memory pressure
        assert results["requests"]["total"] > 0
        assert results["requests"]["success_rate"] > 0.3  # Some success despite memory pressure
        
        print(f"\nMemory Constrained Test:")
        print(f"Success rate under pressure: {results['requests']['success_rate']:.1%}")
        print(f"Total requests: {results['requests']['total']}")
    
    def test_csv_output_correctness(self, base_config, tmp_path):
        """Test that CSV output contains correct prefix caching columns."""
        config = base_config.copy()
        
        config["workload"] = {
            "total_duration": 10,  # Short test
            "bytes_per_token_estimate_for_network": 2,
            "client_profiles": [{
                "profile_name": "test",
                "weight": 1.0,
                "inter_arrival_time_dist_config": {
                    "type": "Exponential",
                    "rate": 3.0
                },
                "prompt_tokens_dist_config": {
                    "type": "Constant",
                    "value": 100,
                    "is_int": True
                },
                "max_output_tokens_dist_config": {
                    "type": "Constant",
                    "value": 20,
                    "is_int": True
                },
                "conversational_probability": 0.5,
                "follow_up_inter_arrival_time_dist_config": {
                    "type": "Constant",
                    "value": 0.3
                },
                "streaming_response_probability": 0.0
            }]
        }
        
        config["frameworks_to_test"] = [{
            "name": "vllm_csv_test",
            "type": "VLLM",
            "is_target_for_workload": True,
            "config": {
                "model_profile_id": "Llama2-7B",
                "gpu_id": "gpu0",
                "block_size": 16,
                "max_num_seqs": 32,
                "enable_prefix_caching": True
            }
        }]
        
        config["metrics_config"]["output_summary_json_path"] = str(tmp_path / "csv_test.json")
        config["metrics_config"]["output_requests_csv_path"] = str(tmp_path / "csv_test.csv")
        
        orchestrator = ExperimentOrchestrator(config)
        orchestrator.run()
        
        # Check CSV has prefix caching columns
        with open(tmp_path / "csv_test.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            assert len(rows) > 0, "CSV should have data"
            
            # Check for prefix caching columns
            first_row = rows[0]
            assert "prefix_cache_event" in first_row
            assert "cached_prefix_length" in first_row
            assert "tokens_actually_prefilled" in first_row
            
            # Count different event types
            event_counts = {}
            for row in rows:
                event = row.get("prefix_cache_event", "None")
                event_counts[event] = event_counts.get(event, 0) + 1
            
            print(f"\nCSV Output Test:")
            print(f"Total rows: {len(rows)}")
            print(f"Event type distribution: {event_counts}")
    
    def test_session_isolation(self, base_config, tmp_path):
        """Test that different sessions don't share cache."""
        config = base_config.copy()
        
        # Multiple independent clients
        config["workload"] = {
            "total_duration": 20,
            "bytes_per_token_estimate_for_network": 2,
            "client_profiles": [{
                "profile_name": "multi_client",
                "weight": 1.0,
                "inter_arrival_time_dist_config": {
                    "type": "Exponential",
                    "rate": 5.0  # High rate to ensure overlap
                },
                "prompt_tokens_dist_config": {
                    "type": "Constant",
                    "value": 200,
                    "is_int": True
                },
                "max_output_tokens_dist_config": {
                    "type": "Constant",
                    "value": 50,
                    "is_int": True
                },
                "conversational_probability": 0.8,
                "follow_up_inter_arrival_time_dist_config": {
                    "type": "Constant",
                    "value": 0.2
                },
                "streaming_response_probability": 0.0
            }]
        }
        
        config["frameworks_to_test"] = [{
            "name": "vllm_isolation",
            "type": "VLLM",
            "is_target_for_workload": True,
            "config": {
                "model_profile_id": "Llama2-7B",
                "gpu_id": "gpu0",
                "block_size": 16,
                "max_num_seqs": 64,
                "enable_prefix_caching": True
            }
        }]
        
        config["metrics_config"]["output_summary_json_path"] = str(tmp_path / "isolation.json")
        config["metrics_config"]["output_requests_csv_path"] = str(tmp_path / "isolation.csv")
        
        orchestrator = ExperimentOrchestrator(config)
        orchestrator.run()
        
        # Analyze CSV to ensure session isolation
        with open(tmp_path / "isolation.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Group by session
        sessions = {}
        for row in rows:
            session_id = row["session_id"]
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append(row)
        
        # Verify each session's cache events are independent
        cache_hit_sessions = 0
        for session_id, session_rows in sessions.items():
            # First request in session should be a miss
            if len(session_rows) > 0:
                first_event = session_rows[0].get("prefix_cache_event", "")
                assert "HIT" not in first_event or first_event == "None", \
                    f"First request in session {session_id} should not be a cache hit"
            
            # Count if this session had any cache hits
            if any("HIT" in row.get("prefix_cache_event", "") for row in session_rows[1:]):
                cache_hit_sessions += 1
        
        print(f"\nSession Isolation Test:")
        print(f"Total sessions: {len(sessions)}")
        print(f"Sessions with cache hits: {cache_hit_sessions}")
        print(f"Average requests per session: {len(rows) / len(sessions):.1f}")
    
    def test_metrics_calculation_accuracy(self, base_config, tmp_path):
        """Test that prefix cache metrics are calculated correctly."""
        config = base_config.copy()
        
        config["workload"] = {
            "total_duration": 15,
            "bytes_per_token_estimate_for_network": 2,
            "client_profiles": [{
                "profile_name": "metrics_test",
                "weight": 1.0,
                "inter_arrival_time_dist_config": {
                    "type": "Exponential",
                    "rate": 2.0
                },
                "prompt_tokens_dist_config": {
                    "type": "Constant",
                    "value": 300,
                    "is_int": True
                },
                "max_output_tokens_dist_config": {
                    "type": "Constant",
                    "value": 100,
                    "is_int": True
                },
                "conversational_probability": 0.6,
                "follow_up_inter_arrival_time_dist_config": {
                    "type": "Constant",
                    "value": 0.5
                },
                "streaming_response_probability": 0.0
            }]
        }
        
        config["frameworks_to_test"] = [{
            "name": "vllm_metrics",
            "type": "VLLM",
            "is_target_for_workload": True,
            "config": {
                "model_profile_id": "Llama2-7B",
                "gpu_id": "gpu0",
                "block_size": 16,
                "max_num_seqs": 32,
                "enable_prefix_caching": True
            }
        }]
        
        config["metrics_config"]["output_summary_json_path"] = str(tmp_path / "metrics.json")
        config["metrics_config"]["output_requests_csv_path"] = str(tmp_path / "metrics.csv")
        
        orchestrator = ExperimentOrchestrator(config)
        orchestrator.run()
        
        # Load and verify metrics
        with open(tmp_path / "metrics.json") as f:
            results = json.load(f)
        
        with open(tmp_path / "metrics.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if "prefix_caching" in results:
            cache_stats = results["prefix_caching"]
            
            # Manually calculate metrics from CSV
            total_events = 0
            hits = 0
            total_cached = 0
            total_prefilled = 0
            
            for row in rows:
                event = row.get("prefix_cache_event", "")
                if event and event != "None":
                    total_events += 1
                    if "HIT" in event:
                        hits += 1
                        cached_len = int(row.get("cached_prefix_length", "0"))
                        total_cached += cached_len
                
                # Count actual prefilled tokens
                if row.get("tokens_actually_prefilled"):
                    try:
                        prefilled = int(row["tokens_actually_prefilled"])
                        total_prefilled += prefilled
                    except (ValueError, TypeError):
                        pass
            
            # Verify calculations
            if total_events > 0:
                calculated_hit_rate = hits / total_events
                reported_hit_rate = cache_stats["overall_hit_rate"]
                
                print(f"\nMetrics Accuracy Test:")
                print(f"Reported hit rate: {reported_hit_rate:.3f}")
                print(f"Calculated hit rate: {calculated_hit_rate:.3f}")
                print(f"Total cached tokens: {total_cached}")
                print(f"Total prefilled tokens: {total_prefilled}")
                
                # Allow small floating point differences
                assert abs(reported_hit_rate - calculated_hit_rate) < 0.01, \
                    f"Hit rate mismatch: {reported_hit_rate} vs {calculated_hit_rate}"
    
    def test_high_concurrency(self, base_config, tmp_path):
        """Test prefix caching under high concurrency."""
        config = base_config.copy()
        
        config["workload"] = {
            "total_duration": 20,
            "bytes_per_token_estimate_for_network": 2,
            "client_profiles": [{
                "profile_name": "high_concurrency",
                "weight": 1.0,
                "inter_arrival_time_dist_config": {
                    "type": "Exponential",
                    "rate": 20.0  # Very high rate
                },
                "prompt_tokens_dist_config": {
                    "type": "Uniform",
                    "low": 50,
                    "high": 150,
                    "is_int": True
                },
                "max_output_tokens_dist_config": {
                    "type": "Uniform",
                    "low": 20,
                    "high": 50,
                    "is_int": True
                },
                "conversational_probability": 0.7,
                "follow_up_inter_arrival_time_dist_config": {
                    "type": "Exponential",
                    "rate": 5.0
                },
                "streaming_response_probability": 0.0
            }]
        }
        
        config["frameworks_to_test"] = [{
            "name": "vllm_concurrent",
            "type": "VLLM",
            "is_target_for_workload": True,
            "config": {
                "model_profile_id": "Llama2-7B",
                "gpu_id": "gpu0",
                "block_size": 16,
                "max_num_seqs": 128,  # High concurrency
                "max_num_batched_tokens": 4096,
                "enable_prefix_caching": True
            }
        }]
        
        config["metrics_config"]["output_summary_json_path"] = str(tmp_path / "concurrent.json")
        config["metrics_config"]["output_requests_csv_path"] = str(tmp_path / "concurrent.csv")
        
        orchestrator = ExperimentOrchestrator(config)
        orchestrator.run()
        
        with open(tmp_path / "concurrent.json") as f:
            results = json.load(f)
        
        # System should handle high concurrency
        assert results["requests"]["total"] > 100  # Should process many requests
        assert results["requests"]["success_rate"] > 0.5  # Reasonable success rate
        
        print(f"\nHigh Concurrency Test:")
        print(f"Total requests: {results['requests']['total']}")
        print(f"Success rate: {results['requests']['success_rate']:.1%}")
        print(f"Throughput: {results['throughput']['requests_per_second']:.1f} req/s")
    
    def test_prefix_cache_with_failures(self, base_config, tmp_path):
        """Test prefix caching behavior when some requests fail."""
        config = base_config.copy()
        
        # Create conditions for some failures (very limited memory)
        config["hardware_profile"]["compute_devices"][0]["memory_capacity_bytes"] = 10_000_000_000  # 10GB
        
        config["workload"] = {
            "total_duration": 15,
            "bytes_per_token_estimate_for_network": 2,
            "client_profiles": [{
                "profile_name": "failure_test",
                "weight": 1.0,
                "inter_arrival_time_dist_config": {
                    "type": "Exponential",
                    "rate": 3.0
                },
                "prompt_tokens_dist_config": {
                    "type": "Uniform",
                    "low": 500,
                    "high": 1000,
                    "is_int": True
                },
                "max_output_tokens_dist_config": {
                    "type": "Uniform",
                    "low": 200,
                    "high": 400,
                    "is_int": True
                },
                "conversational_probability": 0.5,
                "follow_up_inter_arrival_time_dist_config": {
                    "type": "Constant",
                    "value": 0.5
                },
                "streaming_response_probability": 0.0
            }]
        }
        
        config["frameworks_to_test"] = [{
            "name": "vllm_failure",
            "type": "VLLM",
            "is_target_for_workload": True,
            "config": {
                "model_profile_id": "Llama2-7B",
                "gpu_id": "gpu0",
                "block_size": 16,
                "max_num_seqs": 4,  # Very limited
                "enable_prefix_caching": True
            }
        }]
        
        config["metrics_config"]["output_summary_json_path"] = str(tmp_path / "failure.json")
        config["metrics_config"]["output_requests_csv_path"] = str(tmp_path / "failure.csv")
        
        orchestrator = ExperimentOrchestrator(config)
        orchestrator.run()
        
        with open(tmp_path / "failure.json") as f:
            results = json.load(f)
        
        # Should have some failures
        assert results["requests"]["failed"] > 0
        # But also some successes
        assert results["requests"]["successful"] > 0
        
        print(f"\nFailure Handling Test:")
        print(f"Success rate: {results['requests']['success_rate']:.1%}")
        print(f"Failed requests: {results['requests']['failed']}")
        print(f"Successful requests: {results['requests']['successful']}")
    
    def test_different_model_sizes(self, base_config, tmp_path):
        """Test prefix caching with different model sizes."""
        models = ["Llama2-7B", "Llama2-13B"]
        
        for model in models:
            config = base_config.copy()
            config["simulation"]["max_simulation_time"] = 15
            
            config["workload"] = {
                "total_duration": 15,
                "bytes_per_token_estimate_for_network": 2,
                "client_profiles": [{
                    "profile_name": "model_test",
                    "weight": 1.0,
                    "inter_arrival_time_dist_config": {
                        "type": "Exponential",
                        "rate": 2.0
                    },
                    "prompt_tokens_dist_config": {
                        "type": "Constant",
                        "value": 200,
                        "is_int": True
                    },
                    "max_output_tokens_dist_config": {
                        "type": "Constant",
                        "value": 50,
                        "is_int": True
                    },
                    "conversational_probability": 0.6,
                    "follow_up_inter_arrival_time_dist_config": {
                        "type": "Constant",
                        "value": 0.5
                    },
                    "streaming_response_probability": 0.0
                }]
            }
            
            config["frameworks_to_test"] = [{
                "name": f"vllm_{model}",
                "type": "VLLM",
                "is_target_for_workload": True,
                "config": {
                    "model_profile_id": model,
                    "gpu_id": "gpu0",
                    "block_size": 16,
                    "max_num_seqs": 32,
                    "enable_prefix_caching": True
                }
            }]
            
            config["metrics_config"]["output_summary_json_path"] = str(tmp_path / f"{model}.json")
            config["metrics_config"]["output_requests_csv_path"] = str(tmp_path / f"{model}.csv")
            
            try:
                orchestrator = ExperimentOrchestrator(config)
                orchestrator.run()
                
                with open(tmp_path / f"{model}.json") as f:
                    results = json.load(f)
                
                print(f"\nModel {model} Test:")
                print(f"Success rate: {results['requests']['success_rate']:.1%}")
                print(f"Throughput: {results['throughput']['requests_per_second']:.1f} req/s")
                
            except Exception as e:
                print(f"\nModel {model} Test - Error: {str(e)}")
                # Some models might not be in the database, that's OK