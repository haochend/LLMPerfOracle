"""Unit tests for prefix caching functionality."""

import pytest
import simpy

from llmperforacle.frameworks.vllm_framework import VLLMFramework
from llmperforacle.frameworks.models import SequenceState, SessionCacheInfo
from llmperforacle.workload.models import Request
from llmperforacle.metrics.collector import MetricsCollector


class MockVirtualHardware:
    """Mock virtual hardware for testing."""
    
    def get_device_info(self, device_id):
        return type('DeviceInfo', (), {
            'memory_capacity_bytes': 80_000_000_000,
            'memory_gbps': 2039,
            'peak_tflops': {'fp16': 312, 'int8': 624}
        })()
    
    def allocate_memory(self, device_id, bytes_to_allocate):
        return simpy.Environment().timeout(0)
    
    def free_memory(self, device_id, bytes_to_free):
        return simpy.Environment().timeout(0)


class TestPrefixCaching:
    """Test prefix caching functionality."""
    
    @pytest.fixture
    def setup_framework(self):
        """Set up a vLLM framework with prefix caching enabled."""
        env = simpy.Environment()
        
        # Mock configuration
        config = {
            "gpu_id": "gpu0",
            "block_size": 16,
            "max_num_seqs": 10,
            "enable_prefix_caching": True,
        }
        
        # Mock model profile
        model_profile = {
            "num_layers": 32,
            "kv_cache_bytes_per_token_per_layer": 256,
        }
        
        # Create metrics collector
        metrics_config = {
            "percentiles_to_calculate": [0.5, 0.9, 0.99],
            "warm_up_duration_s": 0
        }
        metrics_collector = MetricsCollector(env, metrics_config)
        
        # Create framework
        framework = VLLMFramework(
            framework_id="test_vllm",
            simpy_env=env,
            framework_specific_config=config,
            virtual_hardware=MockVirtualHardware(),
            metrics_collector=metrics_collector,
            model_profile=model_profile,
        )
        
        return env, framework, metrics_collector
    
    def test_conversational_cache_miss(self, setup_framework):
        """Test that first request in a conversation is a cache miss."""
        env, framework, metrics_collector = setup_framework
        
        # Create first request in a conversation
        request = Request(
            request_id="req1",
            client_id="client1",
            session_id="session1",
            arrival_time=0,
            prompt_num_tokens=100,
            max_output_tokens=50,
            is_conversational_turn=False,
            streaming_response=False,
            user_priority=0
        )
        
        # Log request arrival
        metrics_collector.log_request_arrival(
            request.request_id,
            request.arrival_time,
            request.client_id,
            request.session_id,
            request.prompt_num_tokens,
            request.max_output_tokens
        )
        
        # Check prefix cache
        cached_length, tokens_to_prefill = framework._check_prefix_cache(request)
        
        assert cached_length == 0
        assert tokens_to_prefill == 100
        
        # Check metrics were logged correctly
        entry = metrics_collector.all_request_metrics["req1"]
        assert entry.prefix_cache_event_type == "MISS_FULL"
        assert entry.cached_prefix_length_used == 0
        assert entry.num_tokens_actually_prefilled == 100
    
    def test_conversational_cache_hit(self, setup_framework):
        """Test that follow-up request in a conversation gets cache hit."""
        env, framework, metrics_collector = setup_framework
        
        # Simulate completed first turn
        session_cache = SessionCacheInfo(
            session_id="session1",
            total_tokens_in_cache=150,  # 100 prompt + 50 response
            prompt_part_length=100,
            response_part_length=50,
            associated_sequence_id="req1",
            last_update_time=1.0
        )
        framework.active_sessions_kv_state["session1"] = session_cache
        
        # Create follow-up request (history + new input)
        request = Request(
            request_id="req2",
            client_id="client1",
            session_id="session1",
            arrival_time=2.0,
            prompt_num_tokens=200,  # 150 cached + 50 new
            max_output_tokens=50,
            is_conversational_turn=True,
            streaming_response=False,
            user_priority=0
        )
        
        # Log request arrival
        metrics_collector.log_request_arrival(
            request.request_id,
            request.arrival_time,
            request.client_id,
            request.session_id,
            request.prompt_num_tokens,
            request.max_output_tokens
        )
        
        # Check prefix cache
        cached_length, tokens_to_prefill = framework._check_prefix_cache(request)
        
        assert cached_length == 150
        assert tokens_to_prefill == 50
        
        # Check metrics were logged correctly
        entry = metrics_collector.all_request_metrics["req2"]
        assert entry.prefix_cache_event_type == "CONVERSATIONAL_HIT"
        assert entry.cached_prefix_length_used == 150
        assert entry.num_tokens_actually_prefilled == 50
    
    def test_full_cache_hit(self, setup_framework):
        """Test case where entire prompt is cached."""
        env, framework, metrics_collector = setup_framework
        
        # Simulate completed first turn with exact same length
        session_cache = SessionCacheInfo(
            session_id="session1",
            total_tokens_in_cache=100,
            prompt_part_length=100,
            response_part_length=0,
            associated_sequence_id="req1",
            last_update_time=1.0
        )
        framework.active_sessions_kv_state["session1"] = session_cache
        
        # Create request with same prompt (e.g., retry)
        request = Request(
            request_id="req2",
            client_id="client1",
            session_id="session1",
            arrival_time=2.0,
            prompt_num_tokens=100,  # Exactly matches cached
            max_output_tokens=50,
            is_conversational_turn=True,
            streaming_response=False,
            user_priority=0
        )
        
        # Check prefix cache
        cached_length, tokens_to_prefill = framework._check_prefix_cache(request)
        
        # Should not reuse cache if prompt is not longer
        assert cached_length == 0
        assert tokens_to_prefill == 100
    
    def test_session_cache_update(self, setup_framework):
        """Test that session cache is updated after sequence completion."""
        env, framework, metrics_collector = setup_framework
        
        # Create a completed sequence state
        request = Request(
            request_id="req1",
            client_id="client1",
            session_id="session1",
            arrival_time=0,
            prompt_num_tokens=100,
            max_output_tokens=50,
            is_conversational_turn=False,
            streaming_response=False,
            user_priority=0
        )
        
        seq_state = SequenceState(
            request_id="req1",
            request=request,
            status="COMPLETED",
            prompt_tokens_processed=100,
            output_tokens_generated=45,
            allocated_kv_blocks=[0, 1, 2],
            prompt_tokens_fully_processed=100
        )
        
        # Update session cache
        framework._update_session_cache(seq_state)
        
        # Verify cache was updated
        assert "session1" in framework.active_sessions_kv_state
        cache_info = framework.active_sessions_kv_state["session1"]
        assert cache_info.total_tokens_in_cache == 145
        assert cache_info.prompt_part_length == 100
        assert cache_info.response_part_length == 45
        assert cache_info.associated_sequence_id == "req1"
    
    def test_prefix_cache_disabled(self, setup_framework):
        """Test that prefix caching can be disabled."""
        env, framework, metrics_collector = setup_framework
        
        # Disable prefix caching
        framework.enable_prefix_caching = False
        
        # Add session cache that would normally be used
        session_cache = SessionCacheInfo(
            session_id="session1",
            total_tokens_in_cache=150,
            prompt_part_length=100,
            response_part_length=50,
            associated_sequence_id="req1",
            last_update_time=1.0
        )
        framework.active_sessions_kv_state["session1"] = session_cache
        
        # Create request that would normally hit cache
        request = Request(
            request_id="req2",
            client_id="client1",
            session_id="session1",
            arrival_time=2.0,
            prompt_num_tokens=200,
            max_output_tokens=50,
            is_conversational_turn=True,
            streaming_response=False,
            user_priority=0
        )
        
        # Check prefix cache
        cached_length, tokens_to_prefill = framework._check_prefix_cache(request)
        
        # Should not use cache when disabled
        assert cached_length == 0
        assert tokens_to_prefill == 200