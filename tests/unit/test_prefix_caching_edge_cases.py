"""Edge case tests for prefix caching functionality."""

import pytest
import simpy

from llmperforacle.frameworks.vllm_framework import VLLMFramework
from llmperforacle.frameworks.models import SequenceState, SessionCacheInfo
from llmperforacle.workload.models import Request
from llmperforacle.metrics.collector import MetricsCollector


class MockVirtualHardware:
    """Mock virtual hardware for testing."""
    
    def get_device_info(self, device_id):
        return type('DeviceInfo', (), {'memory_capacity_bytes': 80_000_000_000})()
    
    def allocate_memory(self, device_id, bytes_to_allocate):
        return simpy.Environment().timeout(0)
    
    def free_memory(self, device_id, bytes_to_free):
        return simpy.Environment().timeout(0)
    
    def submit_computation_task(self, device_id, task_desc):
        return simpy.Environment().timeout(0)


class TestPrefixCachingEdgeCases:
    """Test edge cases for prefix caching."""
    
    @pytest.fixture
    def setup_framework(self):
        """Set up a vLLM framework with prefix caching enabled."""
        env = simpy.Environment()
        
        config = {
            "gpu_id": "gpu0",
            "block_size": 16,
            "max_num_seqs": 10,
            "enable_prefix_caching": True,
        }
        
        model_profile = {
            "num_layers": 32,
            "kv_cache_bytes_per_token_per_layer": 256,
            "prefill_op_stats": {
                "flops_per_token": 2e9,
                "memory_bytes_per_token": 1e6
            },
            "decode_op_stats": {
                "flops_per_token": 1e8,
                "memory_bytes_per_token": 5e5
            }
        }
        
        metrics_config = {
            "percentiles_to_calculate": [0.5, 0.9],
            "warm_up_duration_s": 0
        }
        metrics_collector = MetricsCollector(env, metrics_config)
        
        framework = VLLMFramework(
            framework_id="test_vllm",
            simpy_env=env,
            framework_specific_config=config,
            virtual_hardware=MockVirtualHardware(),
            metrics_collector=metrics_collector,
            model_profile=model_profile,
        )
        
        return env, framework, metrics_collector
    
    def test_empty_session_cache(self, setup_framework):
        """Test handling of empty session cache."""
        env, framework, metrics_collector = setup_framework
        
        # Request with conversational turn but no cached session
        request = Request(
            request_id="req1",
            client_id="client1",
            session_id="unknown_session",
            arrival_time=0,
            prompt_num_tokens=100,
            max_output_tokens=50,
            is_conversational_turn=True,
            streaming_response=False,
            user_priority=0
        )
        
        metrics_collector.log_request_arrival(
            request.request_id, request.arrival_time, request.client_id,
            request.session_id, request.prompt_num_tokens, request.max_output_tokens
        )
        
        cached_length, tokens_to_prefill = framework._check_prefix_cache(request)
        
        # Should be treated as cache miss
        assert cached_length == 0
        assert tokens_to_prefill == 100
        assert metrics_collector.all_request_metrics["req1"].prefix_cache_event_type == "MISS_FULL"
    
    def test_zero_length_prompt(self, setup_framework):
        """Test handling of zero-length prompts."""
        env, framework, metrics_collector = setup_framework
        
        request = Request(
            request_id="req1",
            client_id="client1",
            session_id="session1",
            arrival_time=0,
            prompt_num_tokens=0,  # Zero length
            max_output_tokens=50,
            is_conversational_turn=False,
            streaming_response=False,
            user_priority=0
        )
        
        metrics_collector.log_request_arrival(
            request.request_id, request.arrival_time, request.client_id,
            request.session_id, request.prompt_num_tokens, request.max_output_tokens
        )
        
        cached_length, tokens_to_prefill = framework._check_prefix_cache(request)
        
        assert cached_length == 0
        assert tokens_to_prefill == 0
    
    def test_exact_cache_match(self, setup_framework):
        """Test when prompt exactly matches cached content."""
        env, framework, metrics_collector = setup_framework
        
        # Set up session cache
        session_cache = SessionCacheInfo(
            session_id="session1",
            total_tokens_in_cache=100,
            prompt_part_length=100,
            response_part_length=0,
            associated_sequence_id="prev_req",
            last_update_time=0.0
        )
        framework.active_sessions_kv_state["session1"] = session_cache
        
        # Request with exact same length as cache
        request = Request(
            request_id="req1",
            client_id="client1",
            session_id="session1",
            arrival_time=1.0,
            prompt_num_tokens=100,  # Exactly matches cache
            max_output_tokens=50,
            is_conversational_turn=True,
            streaming_response=False,
            user_priority=0
        )
        
        metrics_collector.log_request_arrival(
            request.request_id, request.arrival_time, request.client_id,
            request.session_id, request.prompt_num_tokens, request.max_output_tokens
        )
        
        cached_length, tokens_to_prefill = framework._check_prefix_cache(request)
        
        # Current implementation treats this as unexpected (prompt not longer than cache)
        assert cached_length == 0
        assert tokens_to_prefill == 100
        assert "UNEXPECTED" in metrics_collector.all_request_metrics["req1"].prefix_cache_event_type
    
    def test_very_large_cache(self, setup_framework):
        """Test handling of very large cached contexts."""
        env, framework, metrics_collector = setup_framework
        
        # Large cached context
        session_cache = SessionCacheInfo(
            session_id="session1",
            total_tokens_in_cache=10000,  # Very large
            prompt_part_length=8000,
            response_part_length=2000,
            associated_sequence_id="prev_req",
            last_update_time=0.0
        )
        framework.active_sessions_kv_state["session1"] = session_cache
        
        # Request with even larger prompt
        request = Request(
            request_id="req1",
            client_id="client1",
            session_id="session1",
            arrival_time=1.0,
            prompt_num_tokens=12000,  # 10000 cached + 2000 new
            max_output_tokens=50,
            is_conversational_turn=True,
            streaming_response=False,
            user_priority=0
        )
        
        metrics_collector.log_request_arrival(
            request.request_id, request.arrival_time, request.client_id,
            request.session_id, request.prompt_num_tokens, request.max_output_tokens
        )
        
        cached_length, tokens_to_prefill = framework._check_prefix_cache(request)
        
        assert cached_length == 10000
        assert tokens_to_prefill == 2000
        assert metrics_collector.all_request_metrics["req1"].prefix_cache_event_type == "CONVERSATIONAL_HIT"
    
    def test_session_without_id(self, setup_framework):
        """Test handling of requests without session IDs."""
        env, framework, metrics_collector = setup_framework
        
        request = Request(
            request_id="req1",
            client_id="client1",
            session_id=None,  # No session ID
            arrival_time=0,
            prompt_num_tokens=100,
            max_output_tokens=50,
            is_conversational_turn=False,
            streaming_response=False,
            user_priority=0
        )
        
        metrics_collector.log_request_arrival(
            request.request_id, request.arrival_time, request.client_id,
            request.session_id or "", request.prompt_num_tokens, request.max_output_tokens
        )
        
        cached_length, tokens_to_prefill = framework._check_prefix_cache(request)
        
        # Should handle gracefully
        assert cached_length == 0
        assert tokens_to_prefill == 100
    
    def test_concurrent_session_updates(self, setup_framework):
        """Test that session cache handles concurrent updates correctly."""
        env, framework, metrics_collector = setup_framework
        
        # Create multiple completed sequences for same session
        req1 = Request("req1", "client1", "session1", 0, 100, 50, False, False, 0)
        req2 = Request("req2", "client1", "session1", 1, 200, 50, True, False, 0)
        
        seq_state1 = SequenceState(
            request_id="req1",
            request=req1,
            status="COMPLETED",
            prompt_tokens_processed=100,
            output_tokens_generated=50,
            allocated_kv_blocks=[0, 1],
            prompt_tokens_fully_processed=100
        )
        
        seq_state2 = SequenceState(
            request_id="req2",
            request=req2,
            status="COMPLETED",
            prompt_tokens_processed=200,
            output_tokens_generated=75,
            allocated_kv_blocks=[0, 1, 2],
            prompt_tokens_fully_processed=200
        )
        
        # Update session cache multiple times
        framework._update_session_cache(seq_state1)
        assert framework.active_sessions_kv_state["session1"].total_tokens_in_cache == 150
        
        framework._update_session_cache(seq_state2)
        assert framework.active_sessions_kv_state["session1"].total_tokens_in_cache == 275
        assert framework.active_sessions_kv_state["session1"].associated_sequence_id == "req2"
    
    def test_block_allocation_edge_cases(self, setup_framework):
        """Test KV block allocation edge cases."""
        env, framework, metrics_collector = setup_framework
        
        # Test with block_size boundary
        block_size = framework.block_size  # 16
        
        # Exactly one block
        blocks_needed = framework._calculate_prompt_kv_blocks(block_size)
        assert blocks_needed == 1
        
        # Just over one block
        blocks_needed = framework._calculate_prompt_kv_blocks(block_size + 1)
        assert blocks_needed == 2
        
        # Zero tokens
        blocks_needed = framework._calculate_prompt_kv_blocks(0)
        assert blocks_needed == 0
        
        # Large number
        blocks_needed = framework._calculate_prompt_kv_blocks(1000)
        assert blocks_needed == (1000 + block_size - 1) // block_size