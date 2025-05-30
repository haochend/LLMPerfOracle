"""Unit tests for multi-GPU parallelism features."""

import pytest
import simpy
from unittest.mock import Mock, MagicMock, patch

from llmperforacle.frameworks import AbstractLLMFramework, VLLMFramework, ParallelVLLMFramework
from llmperforacle.hardware import VirtualHardwarePlatform, HardwareResourceProfile
from llmperforacle.metrics import MetricsCollector
from llmperforacle.workload import WorkloadGenerator, Request


class TestAbstractFrameworkParallelism:
    """Test parallelism configuration in AbstractLLMFramework."""
    
    def test_parse_parallelism_config_none(self):
        """Test parsing with no parallelism configuration."""
        simpy_env = simpy.Environment()
        config = {"gpu_id": "gpu0"}
        
        framework = self._create_test_framework(simpy_env, config)
        
        assert framework.parallelism_strategy == "None"
        assert framework.gpu_ids == ["gpu0"]
        assert framework.tp_degree == 1
        assert framework.pp_stages == 1
    
    def test_parse_parallelism_config_tp(self):
        """Test parsing tensor parallelism configuration."""
        simpy_env = simpy.Environment()
        config = {
            "parallelism": {
                "strategy": "TP",
                "tp_degree": 4,
                "gpu_ids": ["gpu0", "gpu1", "gpu2", "gpu3"]
            }
        }
        
        framework = self._create_test_framework(simpy_env, config)
        
        assert framework.parallelism_strategy == "TP"
        assert framework.gpu_ids == ["gpu0", "gpu1", "gpu2", "gpu3"]
        assert framework.tp_degree == 4
        assert framework.pp_stages == 1
        assert len(framework.tp_gpu_groups) == 1
        assert framework.tp_gpu_groups[0] == ["gpu0", "gpu1", "gpu2", "gpu3"]
    
    def test_parse_parallelism_config_pp(self):
        """Test parsing pipeline parallelism configuration."""
        simpy_env = simpy.Environment()
        config = {
            "parallelism": {
                "strategy": "PP",
                "pp_stages": 2,
                "gpu_ids": ["gpu0", "gpu1", "gpu2", "gpu3"]
            }
        }
        
        framework = self._create_test_framework(simpy_env, config)
        
        assert framework.parallelism_strategy == "PP"
        assert framework.pp_stages == 2
        assert len(framework.pp_stage_to_gpus) == 2
        assert framework.pp_stage_to_gpus[0] == ["gpu0", "gpu1"]
        assert framework.pp_stage_to_gpus[1] == ["gpu2", "gpu3"]
        assert framework.stage_layer_ranges[0] == (0, 15)  # 32 layers / 2 stages
        assert framework.stage_layer_ranges[1] == (16, 31)
    
    def test_parse_parallelism_config_tp_pp(self):
        """Test parsing combined TP+PP configuration."""
        simpy_env = simpy.Environment()
        config = {
            "parallelism": {
                "strategy": "TP_PP",
                "tp_degree": 2,
                "pp_stages": 2,
                "gpu_ids": ["gpu0", "gpu1", "gpu2", "gpu3"]
            }
        }
        
        framework = self._create_test_framework(simpy_env, config)
        
        assert framework.parallelism_strategy == "TP_PP"
        assert framework.tp_degree == 2
        assert framework.pp_stages == 2
        assert len(framework.pp_stage_to_gpus) == 2
        assert framework.pp_stage_to_gpus[0] == ["gpu0", "gpu1"]
        assert framework.pp_stage_to_gpus[1] == ["gpu2", "gpu3"]
        assert len(framework.tp_gpu_groups) == 2
    
    def test_invalid_tp_config(self):
        """Test error handling for invalid TP configuration."""
        simpy_env = simpy.Environment()
        config = {
            "parallelism": {
                "strategy": "TP",
                "tp_degree": 4,
                "gpu_ids": ["gpu0", "gpu1"]  # Only 2 GPUs for tp_degree=4
            }
        }
        
        with pytest.raises(ValueError, match="TP requires 4 GPUs but got 2"):
            self._create_test_framework(simpy_env, config)
    
    def test_estimate_collective_time(self):
        """Test collective communication time estimation."""
        simpy_env = simpy.Environment()
        config = {
            "parallelism": {
                "strategy": "TP",
                "tp_degree": 4,
                "gpu_ids": ["gpu0", "gpu1", "gpu2", "gpu3"]
            }
        }
        
        framework = self._create_test_framework(simpy_env, config)
        
        # Test AllReduce collective
        data_size = 1024 * 1024  # 1 MB
        time = framework._estimate_collective_time(
            data_size, "AllReduce", ["gpu0", "gpu1", "gpu2", "gpu3"]
        )
        
        # Should be non-zero for multi-GPU
        assert time > 0
        
        # Test single GPU (no collective needed)
        time_single = framework._estimate_collective_time(
            data_size, "AllReduce", ["gpu0"]
        )
        assert time_single == 0.0
    
    def _create_test_framework(self, simpy_env, config):
        """Helper to create a test framework instance."""
        class TestFramework(AbstractLLMFramework):
            def handle_incoming_request(self, request):
                pass
            
            def processing_loop(self):
                pass
            
            def get_status(self):
                return {}
        
        model_profile = {
            "num_layers": 32,
            "hidden_size": 4096,
            "layer_types": {
                "attention": {"flops_per_token_prefill": 1e9},
                "mlp": {"flops_per_token_prefill": 1e9}
            }
        }
        
        return TestFramework(
            framework_id="test_framework",
            simpy_env=simpy_env,
            framework_specific_config=config,
            virtual_hardware=Mock(),
            metrics_collector=Mock(),
            model_profile=model_profile
        )


class TestVLLMTensorParallelism:
    """Test tensor parallelism implementation in VLLM framework."""
    
    @pytest.fixture
    def setup_vllm_tp(self):
        """Set up VLLM with tensor parallelism."""
        simpy_env = simpy.Environment()
        
        # Mock hardware platform
        hardware = Mock(spec=VirtualHardwarePlatform)
        hardware.get_device_info.return_value = Mock(memory_capacity_bytes=80e9)
        
        # Create mock SimPy processes properly
        hardware.submit_computation_task.side_effect = lambda gpu_id, task_desc: simpy_env.timeout(0.001)
        hardware.submit_network_transfer_task.side_effect = lambda src, dst, size: simpy_env.timeout(0.0001)
        hardware.allocate_memory.side_effect = lambda gpu_id, size: simpy_env.timeout(0)
        
        # Mock metrics collector
        metrics = Mock(spec=MetricsCollector)
        
        # Model profile
        model_profile = {
            "num_layers": 32,
            "hidden_size": 4096,
            "kv_cache_bytes_per_token_per_layer": 16384,
            "prefill_op_stats": {"flops_per_token": 14e9, "memory_bytes_per_token": 28e6},
            "decode_op_stats": {"flops_per_token": 14e9, "memory_bytes_per_token": 56e6},
            "layer_types": {
                "attention": {
                    "flops_per_token_prefill": 437.5e6,
                    "flops_per_token_decode": 437.5e6,
                    "memory_bytes_per_token_prefill": 875e3,
                    "memory_bytes_per_token_decode": 1.75e6,
                    "activation_output_bytes_per_token": 8192,
                    "tp_collective_type": "AllReduce"
                },
                "mlp": {
                    "flops_per_token_prefill": 437.5e6,
                    "flops_per_token_decode": 437.5e6,
                    "memory_bytes_per_token_prefill": 875e3,
                    "memory_bytes_per_token_decode": 1.75e6,
                    "activation_output_bytes_per_token": 8192,
                    "tp_collective_type": "AllReduce"
                }
            }
        }
        
        config = {
            "block_size": 16,
            "max_num_seqs": 256,
            "max_num_batched_tokens": 8192,
            "parallelism": {
                "strategy": "TP",
                "tp_degree": 4,
                "gpu_ids": ["gpu0", "gpu1", "gpu2", "gpu3"]
            }
        }
        
        framework = VLLMFramework(
            framework_id="test_vllm_tp",
            simpy_env=simpy_env,
            framework_specific_config=config,
            virtual_hardware=hardware,
            metrics_collector=metrics,
            model_profile=model_profile
        )
        
        return simpy_env, framework, hardware, metrics
    
    def test_tp_prefill_execution(self, setup_vllm_tp):
        """Test tensor parallel prefill execution."""
        simpy_env, framework, hardware, metrics = setup_vllm_tp
        
        # Create a test request
        request = Request(
            request_id="test_req_1",
            client_id="client_1",
            session_id="session_1",
            arrival_time=0.0,
            prompt_num_tokens=128,
            max_output_tokens=64,
            is_conversational_turn=False,
            streaming_response=False,
            user_priority=0
        )
        
        # Execute TP prefill
        # Note: _execute_tp_prefill returns a Process, not a generator
        tp_process = framework._execute_tp_prefill(request.prompt_num_tokens, request.request_id)
        simpy_env.run(until=1.0)
        
        # Verify computation tasks were submitted to all GPUs
        # Each layer should submit tasks to all 4 GPUs
        assert hardware.submit_computation_task.call_count >= 32 * 4  # 32 layers * 4 GPUs
        
        # Verify network transfers for collectives
        assert hardware.submit_network_transfer_task.call_count > 0
    
    def test_tp_decode_execution(self, setup_vllm_tp):
        """Test tensor parallel decode execution."""
        simpy_env, framework, hardware, metrics = setup_vllm_tp
        
        # Execute TP decode
        def test_process():
            yield from framework._execute_tp_decode(batch_size=8)
        
        simpy_env.process(test_process())
        simpy_env.run(until=1.0)
        
        # Verify computation tasks were submitted
        assert hardware.submit_computation_task.call_count >= 32 * 4  # 32 layers * 4 GPUs
        
        # Verify collective communication
        assert hardware.submit_network_transfer_task.call_count > 0
    
    def test_tp_sharding_calculation(self, setup_vllm_tp):
        """Test computation sharding across tensor parallel GPUs."""
        simpy_env, framework, hardware, metrics = setup_vllm_tp
        
        # Test that FLOPs are divided by TP degree
        base_stats = {
            'flops_per_token_prefill': 1e9,
            'memory_bytes_per_token_prefill': 1e6
        }
        
        def test_process():
            yield from framework._dispatch_tp_shardable_operation(
                base_stats,
                batch_size=1,
                sequence_length=100,
                op_type='attention_prefill',
                tp_collective_type='AllReduce',
                tp_gpu_group=framework.tp_gpu_groups[0],
                request_id='test_op'
            )
        
        simpy_env.process(test_process())
        simpy_env.run(until=1.0)
        
        # Check that tasks were submitted with sharded FLOPs
        calls = hardware.submit_computation_task.call_args_list
        for call in calls:
            task_desc = call[0][1]  # Second argument is task description
            # Each GPU should get 1/4 of the FLOPs (tp_degree=4)
            expected_flops = base_stats['flops_per_token_prefill'] * 100 / 4
            assert task_desc['flops_required_fp16'] == expected_flops


class TestPipelineParallelism:
    """Test pipeline parallelism implementation."""
    
    @pytest.fixture
    def setup_parallel_vllm_pp(self):
        """Set up ParallelVLLM with pipeline parallelism."""
        simpy_env = simpy.Environment()
        
        # Mock hardware platform
        hardware = Mock(spec=VirtualHardwarePlatform)
        hardware.get_device_info.return_value = Mock(memory_capacity_bytes=80e9)
        
        # Create mock SimPy processes properly
        hardware.submit_computation_task.side_effect = lambda gpu_id, task_desc: simpy_env.timeout(0.001)
        hardware.submit_network_transfer_task.side_effect = lambda src, dst, size: simpy_env.timeout(0.0001)
        hardware.allocate_memory.side_effect = lambda gpu_id, size: simpy_env.timeout(0)
        hardware.free_memory.side_effect = lambda gpu_id, size: simpy_env.timeout(0)
        
        # Mock metrics collector
        metrics = Mock(spec=MetricsCollector)
        
        # Model profile
        model_profile = {
            "num_layers": 32,
            "hidden_size": 4096,
            "kv_cache_bytes_per_token_per_layer": 16384,
            "prefill_op_stats": {"flops_per_token": 14e9, "memory_bytes_per_token": 28e6},
            "decode_op_stats": {"flops_per_token": 14e9, "memory_bytes_per_token": 56e6},
            "layer_types": {
                "attention": {
                    "flops_per_token_prefill": 437.5e6,
                    "activation_output_bytes_per_token": 8192
                },
                "mlp": {
                    "flops_per_token_prefill": 437.5e6,
                    "activation_output_bytes_per_token": 8192
                }
            }
        }
        
        config = {
            "block_size": 16,
            "max_num_seqs": 256,
            "max_num_batched_tokens": 8192,
            "parallelism": {
                "strategy": "PP",
                "pp_stages": 2,
                "num_microbatches_per_request": 4,
                "gpu_ids": ["gpu0", "gpu1"]
            }
        }
        
        framework = ParallelVLLMFramework(
            framework_id="test_parallel_vllm_pp",
            simpy_env=simpy_env,
            framework_specific_config=config,
            virtual_hardware=hardware,
            metrics_collector=metrics,
            model_profile=model_profile
        )
        
        return simpy_env, framework, hardware, metrics
    
    def test_pp_initialization(self, setup_parallel_vllm_pp):
        """Test pipeline parallelism initialization."""
        simpy_env, framework, hardware, metrics = setup_parallel_vllm_pp
        
        assert framework.pp_stages == 2
        assert len(framework.stage_input_queues) == 2
        assert framework.stage_layer_ranges[0] == (0, 15)
        assert framework.stage_layer_ranges[1] == (16, 31)
        assert framework.pp_stage_to_gpus[0] == ["gpu0"]
        assert framework.pp_stage_to_gpus[1] == ["gpu1"]
    
    def test_microbatch_creation(self, setup_parallel_vllm_pp):
        """Test microbatch creation."""
        simpy_env, framework, hardware, metrics = setup_parallel_vllm_pp
        
        # Create a test request
        request = Request(
            request_id="test_req_1",
            client_id="client_1",
            session_id="session_1",
            arrival_time=0.0,
            prompt_num_tokens=128,
            max_output_tokens=64,
            is_conversational_turn=False,
            streaming_response=False,
            user_priority=0
        )
        
        # Add request to running sequences
        from llmperforacle.frameworks.models import SequenceState
        seq_state = SequenceState(
            request_id=request.request_id,
            request=request,
            status="WAITING_FOR_PREFILL",
            allocated_kv_blocks=[]
        )
        framework.running_sequences[request.request_id] = seq_state
        
        # Create and dispatch microbatches
        def test_process():
            yield from framework._create_and_dispatch_microbatches(request)
        
        simpy_env.process(test_process())
        simpy_env.run(until=0.001)  # Very short time to avoid processing
        
        # Check that microbatches were created
        assert request.request_id in framework.active_microbatches
        assert len(framework.active_microbatches[request.request_id]) == 4  # num_microbatches
        
        # Verify microbatch properties
        for i, mb in enumerate(framework.active_microbatches[request.request_id]):
            assert mb.request_id == request.request_id
            assert mb.microbatch_idx == i
            assert mb.is_prefill == True
            assert mb.current_stage == 0
    
    def test_pipeline_stage_assignment(self, setup_parallel_vllm_pp):
        """Test pipeline stage layer assignment."""
        simpy_env, framework, hardware, metrics = setup_parallel_vllm_pp
        
        # Verify layers are correctly assigned to stages
        assert framework.stage_layer_ranges[0] == (0, 15)  # First 16 layers
        assert framework.stage_layer_ranges[1] == (16, 31)  # Last 16 layers
        
        # Verify GPU assignment to stages
        assert framework.pp_stage_to_gpus[0] == ["gpu0"]
        assert framework.pp_stage_to_gpus[1] == ["gpu1"]
        
        # Test with a request to verify network transfer happens
        from llmperforacle.frameworks.parallel_vllm_framework import MicrobatchState
        
        microbatch = MicrobatchState(
            request_id="test_req_1",
            microbatch_idx=0,
            current_stage=0,
            data_size_tokens=32,
            is_prefill=True,
            arrival_time_at_stage=0.0
        )
        
        # Just verify the transfer method exists and can be called
        def test_process():
            yield from framework._transfer_to_next_stage(0, microbatch)
        
        # Create the process but don't run long enough for workers to consume
        proc = simpy_env.process(test_process())
        simpy_env.run(until=0.0001)
        
        # Verify network transfer was initiated with correct parameters
        hardware.submit_network_transfer_task.assert_called()
        call_args = hardware.submit_network_transfer_task.call_args[0]
        assert call_args[0] == "gpu0"  # Source GPU
        assert call_args[1] == "gpu1"  # Destination GPU
        assert call_args[2] > 0  # Activation size should be positive


class TestDataParallelism:
    """Test data parallelism in workload generator."""
    
    @pytest.fixture
    def setup_workload_dp(self):
        """Set up workload generator with multiple target frameworks."""
        simpy_env = simpy.Environment()
        
        # Create mock target frameworks
        target_fw1 = simpy.Store(simpy_env)
        target_fw2 = simpy.Store(simpy_env)
        target_fw3 = simpy.Store(simpy_env)
        
        target_frameworks_map = {
            "framework_1": target_fw1,
            "framework_2": target_fw2,
            "framework_3": target_fw3
        }
        
        # Mock hardware and metrics
        hardware = Mock()
        hardware.submit_network_transfer_task.side_effect = lambda src, dst, size: simpy_env.timeout(0)
        
        metrics = Mock()
        
        config = {
            "total_duration": 10,
            "bytes_per_token_estimate_for_network": 2,
            "random_seed": 42,
            "load_balancing_strategy": "round_robin",
            "client_profiles": [
                {
                    "profile_name": "test_profile",
                    "weight": 1.0,
                    "inter_arrival_time_dist_config": {"type": "Constant", "value": 0.1},
                    "prompt_tokens_dist_config": {"type": "Constant", "value": 100},
                    "max_output_tokens_dist_config": {"type": "Constant", "value": 50},
                    "conversational_probability": 0.0,
                    "streaming_response_probability": 0.0
                }
            ]
        }
        
        workload = WorkloadGenerator(
            simpy_env=simpy_env,
            config=config,
            target_frameworks_map=target_frameworks_map,
            metrics_collector=metrics,
            hardware_platform=hardware
        )
        
        return simpy_env, workload, target_frameworks_map, metrics
    
    def test_round_robin_load_balancing(self, setup_workload_dp):
        """Test round-robin load balancing across frameworks."""
        simpy_env, workload, target_frameworks_map, metrics = setup_workload_dp
        
        # Generate several requests
        for i in range(9):
            request = Request(
                request_id=f"req_{i}",
                client_id="client_1",
                session_id=f"session_{i}",
                arrival_time=i * 0.1,
                prompt_num_tokens=100,
                max_output_tokens=50,
                is_conversational_turn=False,
                streaming_response=False,
                user_priority=0
            )
            workload.request_counter = i
            target_fw_id = workload._select_target_framework(request)
            
            # Should distribute evenly: 0->fw1, 1->fw2, 2->fw3, 3->fw1, etc.
            expected_fw = f"framework_{(i % 3) + 1}"
            assert target_fw_id == expected_fw
        
        # Check request counts
        assert workload.framework_request_counts["framework_1"] == 3
        assert workload.framework_request_counts["framework_2"] == 3
        assert workload.framework_request_counts["framework_3"] == 3
    
    def test_least_loaded_balancing(self, setup_workload_dp):
        """Test least-loaded load balancing strategy."""
        simpy_env, workload, target_frameworks_map, metrics = setup_workload_dp
        
        workload.load_balancing_strategy = "least_loaded"
        
        # Pre-populate some queues
        target_frameworks_map["framework_1"].put("dummy1")
        target_frameworks_map["framework_1"].put("dummy2")
        target_frameworks_map["framework_2"].put("dummy1")
        # framework_3 is empty
        
        request = Request(
            request_id="test_req",
            client_id="client_1",
            session_id="session_1",
            arrival_time=0.0,
            prompt_num_tokens=100,
            max_output_tokens=50,
            is_conversational_turn=False,
            streaming_response=False,
            user_priority=0
        )
        
        # Should select framework_3 as it has the smallest queue
        target_fw_id = workload._select_target_framework(request)
        assert target_fw_id == "framework_3"
    
    def test_session_affinity_balancing(self, setup_workload_dp):
        """Test session affinity load balancing."""
        simpy_env, workload, target_frameworks_map, metrics = setup_workload_dp
        
        workload.load_balancing_strategy = "session_affinity"
        
        # First request in a session
        request1 = Request(
            request_id="req_1",
            client_id="client_1",
            session_id="session_1",
            arrival_time=0.0,
            prompt_num_tokens=100,
            max_output_tokens=50,
            is_conversational_turn=False,
            streaming_response=False,
            user_priority=0
        )
        
        workload.request_counter = 0
        target_fw1 = workload._select_target_framework(request1)
        
        # Initialize the session mapping if it doesn't exist
        if not hasattr(workload, '_session_fw_mapping'):
            workload._session_fw_mapping = {}
        workload._session_fw_mapping["session_1"] = target_fw1
        
        # Conversational turn in same session
        request2 = Request(
            request_id="req_2",
            client_id="client_1",
            session_id="session_1",
            arrival_time=1.0,
            prompt_num_tokens=50,
            max_output_tokens=50,
            is_conversational_turn=True,
            streaming_response=False,
            user_priority=0
        )
        
        workload.request_counter = 1
        target_fw2 = workload._select_target_framework(request2)
        
        # Should route to same framework
        assert target_fw1 == target_fw2
    
    def test_random_load_balancing(self, setup_workload_dp):
        """Test random load balancing strategy."""
        simpy_env, workload, target_frameworks_map, metrics = setup_workload_dp
        
        workload.load_balancing_strategy = "random"
        
        # Generate multiple requests
        selected_frameworks = set()
        for i in range(20):
            request = Request(
                request_id=f"req_{i}",
                client_id="client_1",
                session_id=f"session_{i}",
                arrival_time=i * 0.1,
                prompt_num_tokens=100,
                max_output_tokens=50,
                is_conversational_turn=False,
                streaming_response=False,
                user_priority=0
            )
            
            target_fw = workload._select_target_framework(request)
            selected_frameworks.add(target_fw)
        
        # With 20 requests, should have selected all 3 frameworks at least once
        assert len(selected_frameworks) == 3


class TestCombinedParallelism:
    """Test combined TP+PP parallelism."""
    
    def test_tp_pp_configuration(self):
        """Test combined TP+PP configuration parsing."""
        simpy_env = simpy.Environment()
        
        # Mock components
        hardware = Mock()
        hardware.get_device_info.return_value = Mock(memory_capacity_bytes=80e9)
        metrics = Mock()
        
        model_profile = {
            "num_layers": 32,
            "hidden_size": 4096,
            "kv_cache_bytes_per_token_per_layer": 16384
        }
        
        config = {
            "block_size": 16,
            "parallelism": {
                "strategy": "TP_PP",
                "tp_degree": 2,
                "pp_stages": 2,
                "gpu_ids": ["gpu0", "gpu1", "gpu2", "gpu3"]
            }
        }
        
        framework = ParallelVLLMFramework(
            framework_id="test_tp_pp",
            simpy_env=simpy_env,
            framework_specific_config=config,
            virtual_hardware=hardware,
            metrics_collector=metrics,
            model_profile=model_profile
        )
        
        # Check configuration
        assert framework.parallelism_strategy == "TP_PP"
        assert framework.tp_degree == 2
        assert framework.pp_stages == 2
        
        # Check GPU assignment
        # Stage 0: GPUs 0,1 (TP group)
        # Stage 1: GPUs 2,3 (TP group)
        assert framework.pp_stage_to_gpus[0] == ["gpu0", "gpu1"]
        assert framework.pp_stage_to_gpus[1] == ["gpu2", "gpu3"]
        assert len(framework.tp_gpu_groups) == 2
        assert framework.tp_gpu_groups[0] == ["gpu0", "gpu1"]
        assert framework.tp_gpu_groups[1] == ["gpu2", "gpu3"]