"""Unit tests for cross-request prefix caching functionality."""

import pytest
import simpy
from typing import List

from llmperforacle.frameworks.vllm_framework import VLLMFramework
from llmperforacle.frameworks.models import SequenceState, GlobalPrefixCacheInfo
from llmperforacle.frameworks.prefix_utils import hash_token_sequence, find_longest_prefix_match
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


class TestCrossRequestCaching:
    """Test cross-request prefix caching functionality."""
    
    @pytest.fixture
    def setup_framework(self):
        """Set up a vLLM framework with cross-request caching enabled."""
        env = simpy.Environment()
        
        config = {
            "gpu_id": "gpu0",
            "block_size": 16,
            "max_num_seqs": 10,
            "enable_prefix_caching": True,
            "enable_cross_request_caching": True,
            "min_prefix_cache_length": 50,
            "max_prefix_cache_size": 5,
            "prefix_eviction_policy": "lru",
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
    
    def test_cross_request_cache_miss(self, setup_framework):
        """Test that first request with a prefix is a cache miss."""
        env, framework, metrics_collector = setup_framework
        
        # Create request with specific tokens
        tokens = list(range(1000, 1100))  # 100 tokens
        request = Request(
            request_id="req1",
            client_id="client1",
            session_id=None,  # No session (cross-request)
            arrival_time=0,
            prompt_num_tokens=100,
            max_output_tokens=50,
            is_conversational_turn=False,
            streaming_response=False,
            user_priority=0,
            prompt_tokens=tokens
        )
        
        metrics_collector.log_request_arrival(
            request.request_id, request.arrival_time, request.client_id,
            request.session_id or "", request.prompt_num_tokens, request.max_output_tokens
        )
        
        cached_length, tokens_to_prefill = framework._check_prefix_cache(request)
        
        assert cached_length == 0
        assert tokens_to_prefill == 100
        assert metrics_collector.all_request_metrics["req1"].prefix_cache_event_type == "MISS_FULL"
    
    def test_cross_request_cache_population(self, setup_framework):
        """Test that prefixes are added to global cache after processing."""
        env, framework, metrics_collector = setup_framework
        
        # Create request with long prefix
        tokens = list(range(1000, 1200))  # 200 tokens
        request = Request(
            request_id="req1",
            client_id="client1",
            session_id=None,
            arrival_time=0,
            prompt_num_tokens=200,
            max_output_tokens=50,
            is_conversational_turn=False,
            streaming_response=False,
            user_priority=0,
            prompt_tokens=tokens
        )
        
        # Create sequence state
        seq_state = SequenceState(
            request_id="req1",
            request=request,
            status="COMPLETED",
            prompt_tokens_processed=200,
            output_tokens_generated=50,
            allocated_kv_blocks=[0, 1, 2],
            cached_prefix_length_used=0,  # No cache used
            num_tokens_requiring_prefill=200,
            prompt_tokens_fully_processed=200
        )
        
        # Should add prefix to global cache
        framework._maybe_add_to_global_cache(seq_state)
        
        # Check that prefix was added (50 or 100 token prefix)
        assert len(framework.global_prefix_store) > 0
        
        # Verify the cached prefix
        prefix_hash_50 = hash_token_sequence(tokens, 50)
        prefix_hash_100 = hash_token_sequence(tokens, 100)
        
        assert prefix_hash_50 in framework.global_prefix_store or prefix_hash_100 in framework.global_prefix_store
    
    def test_cross_request_cache_hit(self, setup_framework):
        """Test cache hit for subsequent request with same prefix."""
        env, framework, metrics_collector = setup_framework
        
        # Manually populate global cache
        common_prefix = list(range(1000, 1100))  # 100 tokens
        prefix_hash = hash_token_sequence(common_prefix, 100)
        
        framework.global_prefix_store[prefix_hash] = GlobalPrefixCacheInfo(
            prefix_hash=prefix_hash,
            prefix_length=100,
            kv_block_ids=[],
            reference_count=0,
            last_access_time=0,
            creation_time=0,
            access_count=0,
            prefix_tokens=common_prefix
        )
        
        # Create request with same prefix plus additional tokens
        tokens = common_prefix + list(range(2000, 2050))  # 100 + 50 = 150 tokens
        request = Request(
            request_id="req2",
            client_id="client2",
            session_id=None,
            arrival_time=1.0,
            prompt_num_tokens=150,
            max_output_tokens=50,
            is_conversational_turn=False,
            streaming_response=False,
            user_priority=0,
            prompt_tokens=tokens
        )
        
        metrics_collector.log_request_arrival(
            request.request_id, request.arrival_time, request.client_id,
            request.session_id or "", request.prompt_num_tokens, request.max_output_tokens
        )
        
        cached_length, tokens_to_prefill = framework._check_prefix_cache(request)
        
        assert cached_length == 100
        assert tokens_to_prefill == 50
        assert metrics_collector.all_request_metrics["req2"].prefix_cache_event_type == "CROSS_REQUEST_HIT"
        
        # Check that reference count and access count were updated
        assert framework.global_prefix_store[prefix_hash].reference_count == 1
        assert framework.global_prefix_store[prefix_hash].access_count == 1
    
    def test_lru_eviction(self, setup_framework):
        """Test LRU eviction when cache is full."""
        env, framework, metrics_collector = setup_framework
        
        # Fill cache to capacity (max_size = 5)
        for i in range(5):
            prefix = list(range(1000 + i*100, 1000 + (i+1)*100))
            prefix_hash = hash_token_sequence(prefix, 100)
            framework.global_prefix_store[prefix_hash] = GlobalPrefixCacheInfo(
                prefix_hash=prefix_hash,
                prefix_length=100,
                kv_block_ids=[],
                reference_count=0,
                last_access_time=i,  # Different access times
                creation_time=0,
                access_count=i,
                prefix_tokens=prefix
            )
        
        assert len(framework.global_prefix_store) == 5
        
        # Add one more - should trigger eviction
        new_prefix = list(range(2000, 2100))
        new_hash = hash_token_sequence(new_prefix, 100)
        
        # Create sequence state to trigger cache addition
        request = Request(
            request_id="req_new",
            client_id="client1",
            session_id=None,
            arrival_time=10,
            prompt_num_tokens=150,
            max_output_tokens=50,
            is_conversational_turn=False,
            streaming_response=False,
            user_priority=0,
            prompt_tokens=new_prefix + list(range(3000, 3050))
        )
        
        seq_state = SequenceState(
            request_id="req_new",
            request=request,
            status="COMPLETED",
            prompt_tokens_processed=150,
            output_tokens_generated=50,
            allocated_kv_blocks=[],
            cached_prefix_length_used=0,
            num_tokens_requiring_prefill=150,
            prompt_tokens_fully_processed=150
        )
        
        framework._add_prefix_to_global_cache(new_hash, 100, new_prefix, seq_state)
        
        # Should still have 5 entries (one was evicted)
        assert len(framework.global_prefix_store) == 5
        assert new_hash in framework.global_prefix_store
        
        # The entry with access_time=0 should have been evicted
        oldest_hash = hash_token_sequence(list(range(1000, 1100)), 100)
        assert oldest_hash not in framework.global_prefix_store
    
    def test_no_eviction_with_active_references(self, setup_framework):
        """Test that entries with active references are not evicted."""
        env, framework, metrics_collector = setup_framework
        
        # Fill cache with entries, one having active reference
        for i in range(5):
            prefix = list(range(1000 + i*100, 1000 + (i+1)*100))
            prefix_hash = hash_token_sequence(prefix, 100)
            framework.global_prefix_store[prefix_hash] = GlobalPrefixCacheInfo(
                prefix_hash=prefix_hash,
                prefix_length=100,
                kv_block_ids=[],
                reference_count=1 if i == 0 else 0,  # First has active reference
                last_access_time=i,
                creation_time=0,
                access_count=i,
                prefix_tokens=prefix
            )
        
        # Try eviction
        framework._evict_from_global_cache()
        
        # The entry with reference_count=1 should not be evicted
        protected_hash = hash_token_sequence(list(range(1000, 1100)), 100)
        assert protected_hash in framework.global_prefix_store
        
        # But another entry should have been evicted
        assert len(framework.global_prefix_store) == 4
    
    def test_prefix_hash_utility(self):
        """Test the prefix hashing utility functions."""
        tokens = list(range(1000, 1200))
        
        # Test basic hashing
        hash1 = hash_token_sequence(tokens, 100)
        hash2 = hash_token_sequence(tokens, 100)
        hash3 = hash_token_sequence(tokens, 150)
        
        assert hash1 == hash2  # Same tokens, same length
        assert hash1 != hash3  # Different lengths
        
        # Test find_longest_prefix_match
        prefix_store = {
            hash_token_sequence(tokens, 50): GlobalPrefixCacheInfo(
                prefix_hash="hash50", prefix_length=50, kv_block_ids=[]
            ),
            hash_token_sequence(tokens, 100): GlobalPrefixCacheInfo(
                prefix_hash="hash100", prefix_length=100, kv_block_ids=[]
            ),
        }
        
        # Should find the longest match (100)
        matching_hash, length = find_longest_prefix_match(
            tokens[:150], prefix_store, min_prefix_length=50
        )
        assert length == 100
        
        # No match with different tokens
        different_tokens = list(range(2000, 2150))
        matching_hash, length = find_longest_prefix_match(
            different_tokens, prefix_store, min_prefix_length=50
        )
        assert matching_hash is None
        assert length == 0