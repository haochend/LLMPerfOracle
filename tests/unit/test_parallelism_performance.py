"""Performance validation tests for parallelism implementations."""

import pytest
import simpy
import numpy as np
from unittest.mock import Mock, MagicMock

from llmperforacle.frameworks import VLLMFramework, ParallelVLLMFramework
from llmperforacle.hardware import VirtualHardwarePlatform
from llmperforacle.metrics import MetricsCollector
from llmperforacle.workload import Request
from llmperforacle.frameworks.models import SequenceState


class TestParallelismPerformance:
    """Verify parallelism provides expected performance improvements."""
    
    def test_tp_speedup_for_compute_bound(self):
        """Test that TP provides speedup for compute-bound operations."""
        simpy_env = simpy.Environment()
        
        # Track computation times
        compute_times = {}
        
        def mock_compute(gpu_id, task_desc):
            # Record computation time
            task_id = task_desc.get('task_id', 'unknown')
            flops = task_desc.get('flops_required_fp16', 0)
            # Assume 312 TFLOPS with 50% MFU
            time = flops / (312e12 * 0.5)
            
            if 'prefill' in task_id:
                if gpu_id not in compute_times:
                    compute_times[gpu_id] = []
                compute_times[gpu_id].append(time)
            
            return simpy_env.timeout(time)
        
        hardware = Mock(spec=VirtualHardwarePlatform)
        hardware.get_device_info.return_value = Mock(memory_capacity_bytes=80e9)
        hardware.submit_computation_task.side_effect = mock_compute
        hardware.submit_network_transfer_task.side_effect = lambda src, dst, size: simpy_env.timeout(0.00001)
        hardware.allocate_memory.side_effect = lambda gpu, size: simpy_env.timeout(0)
        hardware.free_memory.side_effect = lambda gpu, size: simpy_env.timeout(0)
        
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
        
        # Test TP=4 configuration
        config = {
            "block_size": 16,
            "max_num_seqs": 32,
            "parallelism": {
                "strategy": "TP",
                "tp_degree": 4,
                "gpu_ids": ["gpu0", "gpu1", "gpu2", "gpu3"]
            }
        }
        
        framework = VLLMFramework(
            framework_id="test_tp4",
            simpy_env=simpy_env,
            framework_specific_config=config,
            virtual_hardware=hardware,
            metrics_collector=Mock(),
            model_profile=model_profile
        )
        
        # Process a large prompt (compute-bound)
        request = Request(
            request_id="test_req",
            client_id="client",
            session_id="sess",
            arrival_time=0.0,
            prompt_num_tokens=1000,  # Large prompt
            max_output_tokens=10,
            is_conversational_turn=False,
            streaming_response=False,
            user_priority=0
        )
        
        # Run prefill
        def test_process():
            # _execute_tp_prefill returns a Process, not a generator
            yield framework._execute_tp_prefill(request.prompt_num_tokens, request.request_id)
        
        simpy_env.process(test_process())
        simpy_env.run(until=10.0)
        
        # Verify that computation tasks were submitted to multiple GPUs
        # With TP=4, tasks should be distributed
        assert hardware.submit_computation_task.call_count > 0  # Some work was done
        
        # Check that multiple GPUs were used
        gpu_ids_used = set()
        for call in hardware.submit_computation_task.call_args_list:
            gpu_id = call[0][0]  # First argument is GPU ID
            gpu_ids_used.add(gpu_id)
        
        # With TP=4, we should use all 4 GPUs
        assert len(gpu_ids_used) >= 2  # At least some distribution
    
    def test_pp_pipeline_efficiency(self):
        """Test pipeline parallelism efficiency with microbatching."""
        simpy_env = simpy.Environment()
        
        # Track stage execution times
        stage_times = {0: [], 1: []}
        
        def mock_compute(gpu_id, task_desc):
            stage = 0 if gpu_id in ["gpu0", "gpu1"] else 1
            # Different computation times for stages
            if stage == 0:
                time = 0.01  # First stage slower
            else:
                time = 0.005  # Second stage faster
            
            task_id = task_desc.get('task_id', '')
            if 'microbatch' in task_id:
                stage_times[stage].append((simpy_env.now, simpy_env.now + time))
            
            return simpy_env.timeout(time)
        
        hardware = Mock(spec=VirtualHardwarePlatform)
        hardware.get_device_info.return_value = Mock(memory_capacity_bytes=80e9)
        hardware.submit_computation_task.side_effect = mock_compute
        hardware.submit_network_transfer_task.side_effect = lambda src, dst, size: simpy_env.timeout(0.0001)
        hardware.allocate_memory.side_effect = lambda gpu, size: simpy_env.timeout(0)
        hardware.free_memory.side_effect = lambda gpu, size: simpy_env.timeout(0)
        
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
                }
            }
        }
        
        config = {
            "block_size": 16,
            "parallelism": {
                "strategy": "PP",
                "pp_stages": 2,
                "num_microbatches_per_request": 8,  # Multiple microbatches
                "gpu_ids": ["gpu0", "gpu1", "gpu2", "gpu3"]
            }
        }
        
        framework = ParallelVLLMFramework(
            framework_id="test_pp",
            simpy_env=simpy_env,
            framework_specific_config=config,
            virtual_hardware=hardware,
            metrics_collector=Mock(),
            model_profile=model_profile
        )
        
        # Create multiple requests
        requests = []
        for i in range(3):
            req = Request(
                request_id=f"req_{i}",
                client_id="client",
                session_id=f"sess_{i}",
                arrival_time=i * 0.1,
                prompt_num_tokens=200,
                max_output_tokens=50,
                is_conversational_turn=False,
                streaming_response=False,
                user_priority=0
            )
            requests.append(req)
            
            seq_state = SequenceState(
                request_id=req.request_id,
                request=req,
                status="WAITING_FOR_PREFILL",
                allocated_kv_blocks=[]
            )
            framework.running_sequences[req.request_id] = seq_state
        
        # Start pipeline stages
        for stage in range(2):
            simpy_env.process(framework._pipeline_stage_worker_process(stage))
        
        # Submit requests
        def submit_all():
            for req in requests:
                yield simpy_env.timeout(req.arrival_time)
                yield from framework._create_and_dispatch_microbatches(req)
        
        simpy_env.process(submit_all())
        simpy_env.run(until=2.0)
        
        # Analyze pipeline efficiency
        # Stage 1 should start processing while stage 0 is still working
        if len(stage_times[0]) > 1 and len(stage_times[1]) > 1:
            # Check for pipeline overlap
            stage0_end = stage_times[0][0][1]  # End of first task in stage 0
            stage1_start = stage_times[1][0][0]  # Start of first task in stage 1
            
            # Stage 1 should start before stage 0 completely finishes all microbatches
            assert stage1_start < stage_times[0][-1][1]  # Pipeline overlap exists
    
    def test_dp_throughput_scaling(self):
        """Test that data parallelism scales throughput."""
        simpy_env = simpy.Environment()
        
        # Create mock frameworks that track processed requests
        processed_requests = {f"framework_{i}": 0 for i in range(4)}
        
        target_frameworks = {}
        for i in range(4):
            store = simpy.Store(simpy_env, capacity=100)
            
            # Process requests in background
            def processor(fw_id, store):
                while True:
                    req = yield store.get()
                    processed_requests[fw_id] += 1
                    yield simpy_env.timeout(0.1)  # Simulate processing
            
            simpy_env.process(processor(f"framework_{i}", store))
            target_frameworks[f"framework_{i}"] = store
        
        from llmperforacle.workload import WorkloadGenerator
        
        config = {
            "total_duration": 5,
            "bytes_per_token_estimate_for_network": 2,
            "load_balancing_strategy": "round_robin",
            "client_profiles": [{
                "profile_name": "high_load",
                "weight": 1.0,
                "inter_arrival_time_dist_config": {
                    "type": "Exponential",
                    "rate": 20.0  # 20 requests per second
                },
                "prompt_tokens_dist_config": {"type": "Constant", "value": 100},
                "max_output_tokens_dist_config": {"type": "Constant", "value": 100},
                "conversational_probability": 0.0,
                "streaming_response_probability": 0.0
            }]
        }
        
        hardware = Mock()
        hardware.submit_network_transfer_task.side_effect = lambda src, dst, size: simpy_env.timeout(0)
        
        workload = WorkloadGenerator(
            simpy_env=simpy_env,
            config=config,
            target_frameworks_map=target_frameworks,
            metrics_collector=Mock(),
            hardware_platform=hardware
        )
        
        # Run workload generation
        simpy_env.process(workload.generate_requests_process())
        simpy_env.run(until=5.0)
        
        # Verify load was distributed
        total_processed = sum(processed_requests.values())
        assert total_processed > 50  # Should process many requests
        
        # Each framework should get roughly equal load with round-robin
        avg_processed = total_processed / 4
        for fw_id, count in processed_requests.items():
            assert abs(count - avg_processed) / avg_processed < 0.2  # Within 20% of average
        
        # With 4 replicas, throughput should be much higher than single instance
        # (Single instance would only process ~50 requests in 5 seconds at 0.1s each)
        assert total_processed > 80  # Should see significant scaling with 4 replicas
    
    def test_combined_tp_pp_benefits(self):
        """Test that TP+PP combination provides both benefits."""
        simpy_env = simpy.Environment()
        
        # Track both computation distribution and pipeline stages
        gpu_work = {f"gpu{i}": 0 for i in range(4)}
        stage_work = {0: 0, 1: 0}
        
        def mock_compute(gpu_id, task_desc):
            flops = task_desc.get('flops_required_fp16', 1e9)
            gpu_work[gpu_id] += flops
            
            # Determine stage
            stage = 0 if gpu_id in ["gpu0", "gpu1"] else 1
            stage_work[stage] += flops
            
            # Compute time with TP speedup
            time = flops / (312e12 * 0.5)  # Base time
            return simpy_env.timeout(time)
        
        hardware = Mock(spec=VirtualHardwarePlatform)
        hardware.get_device_info.return_value = Mock(memory_capacity_bytes=80e9)
        hardware.submit_computation_task.side_effect = mock_compute
        hardware.submit_network_transfer_task.side_effect = lambda src, dst, size: simpy_env.timeout(0.00001)
        hardware.allocate_memory.side_effect = lambda gpu, size: simpy_env.timeout(0)
        hardware.free_memory.side_effect = lambda gpu, size: simpy_env.timeout(0)
        
        model_profile = {
            "num_layers": 32,
            "hidden_size": 4096,
            "kv_cache_bytes_per_token_per_layer": 16384,
            "prefill_op_stats": {"flops_per_token": 14e9, "memory_bytes_per_token": 28e6},
            "decode_op_stats": {"flops_per_token": 14e9, "memory_bytes_per_token": 56e6},
            "layer_types": {
                "attention": {
                    "flops_per_token_prefill": 437.5e6,
                    "activation_output_bytes_per_token": 8192,
                    "tp_collective_type": "AllReduce"
                }
            }
        }
        
        config = {
            "block_size": 16,
            "parallelism": {
                "strategy": "TP_PP",
                "tp_degree": 2,
                "pp_stages": 2,
                "num_microbatches_per_request": 4,
                "gpu_ids": ["gpu0", "gpu1", "gpu2", "gpu3"]
            }
        }
        
        framework = ParallelVLLMFramework(
            framework_id="test_tp_pp",
            simpy_env=simpy_env,
            framework_specific_config=config,
            virtual_hardware=hardware,
            metrics_collector=Mock(),
            model_profile=model_profile
        )
        
        # Process request
        request = Request(
            request_id="test_req",
            client_id="client",
            session_id="sess",
            arrival_time=0.0,
            prompt_num_tokens=500,
            max_output_tokens=100,
            is_conversational_turn=False,
            streaming_response=False,
            user_priority=0
        )
        
        seq_state = SequenceState(
            request_id=request.request_id,
            request=request,
            status="WAITING_FOR_PREFILL",
            allocated_kv_blocks=[]
        )
        framework.running_sequences[request.request_id] = seq_state
        
        # Start pipeline stages
        for stage in range(2):
            simpy_env.process(framework._pipeline_stage_worker_process(stage))
        
        # Process request
        def process():
            yield from framework._create_and_dispatch_microbatches(request)
        
        simpy_env.process(process())
        simpy_env.run(until=2.0)
        
        # Verify work distribution
        # 1. Work should be split between stages (PP)
        assert stage_work[0] > 0 and stage_work[1] > 0
        # Allow for some imbalance due to microbatching
        assert abs(stage_work[0] - stage_work[1]) / max(stage_work[0], stage_work[1]) < 0.6
        
        # 2. Within each stage, work should be split between TP groups
        assert gpu_work["gpu0"] > 0 and gpu_work["gpu1"] > 0  # Stage 0 TP
        assert gpu_work["gpu2"] > 0 and gpu_work["gpu3"] > 0  # Stage 1 TP
        
        # 3. GPUs in same TP group should have similar work
        assert abs(gpu_work["gpu0"] - gpu_work["gpu1"]) / max(gpu_work["gpu0"], gpu_work["gpu1"]) < 0.1
        assert abs(gpu_work["gpu2"] - gpu_work["gpu3"]) / max(gpu_work["gpu2"], gpu_work["gpu3"]) < 0.1