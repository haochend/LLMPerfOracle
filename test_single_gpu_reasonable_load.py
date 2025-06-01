#!/usr/bin/env python3
"""Test Single GPU with same reasonable load as TP=8 for fair comparison."""

import os
import sys
import json
import yaml
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llmperforacle.orchestration.experiment_orchestrator import ExperimentOrchestrator

def main():
    """Run Single GPU with reasonable request rates."""
    print("Single GPU Heavy Workload Test - Reasonable Load")
    print("Using same request rate as TP=8 test for fair comparison")
    
    # Load Single GPU configuration
    config_file = "configs/gh200_experiment_single.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # 30-second simulation (same as TP=8)
    config['simulation']['max_simulation_time'] = 30
    config['workload']['total_duration'] = 30
    config['metrics_config']['warm_up_duration_s'] = 5
    
    # Calculate current total rate
    total_rate = 0
    for profile in config['workload']['client_profiles']:
        weight = profile.get('weight', 1.0)
        rate = profile['inter_arrival_time_dist_config'].get('rate', 0)
        total_rate += weight * rate
    
    print(f"\nOriginal total request rate: {total_rate:.1f} req/s")
    
    # Scale down to same rate as TP=8 test
    target_total_rate = 1.5  # requests per second
    scale_factor = target_total_rate / total_rate
    
    print(f"Scaling down by factor of {scale_factor:.3f}")
    print(f"New total request rate: {target_total_rate} req/s (same as TP=8 test)")
    
    # Apply scaling
    for profile in config['workload']['client_profiles']:
        if 'inter_arrival_time_dist_config' in profile:
            if profile['inter_arrival_time_dist_config']['type'] == 'Exponential':
                old_rate = profile['inter_arrival_time_dist_config']['rate']
                new_rate = old_rate * scale_factor
                profile['inter_arrival_time_dist_config']['rate'] = new_rate
    
    # Enable chunked prefill for single GPU too
    for framework in config.get('frameworks_to_test', []):
        if 'config' not in framework:
            framework['config'] = {}
        framework['config']['enable_chunked_prefill'] = True
        framework['config']['prefill_chunk_size'] = 4096
        framework['config'].pop('max_num_batched_tokens', None)
    
    print("\nStarting simulation...")
    
    try:
        start_time = time.time()
        
        # Run simulation
        orchestrator = ExperimentOrchestrator(config)
        orchestrator.setup_simulation()
        
        framework = orchestrator.llm_framework_instances[0]
        print(f"\nFramework: Single GPU")
        print(f"GPU: {framework.gpu_id}")
        print(f"Max batched tokens: {framework.max_batched_tokens_per_iteration}")
        print(f"Chunked prefill: {framework.enable_chunked_prefill}")
        
        results = orchestrator.run()
        
        elapsed = time.time() - start_time
        
        print(f"\n✓ Completed in {elapsed:.1f}s real time!")
        print(f"\n{'='*60}")
        print("RESULTS - Single GPU with Heavy Workload")
        print(f"{'='*60}")
        
        print(f"\nRequests:")
        print(f"  Total: {results['requests']['total']}")
        print(f"  Successful: {results['requests']['successful']}")
        print(f"  Failed: {results['requests']['failed']}")
        print(f"  Success rate: {results['requests']['success_rate']:.1%}")
        
        if results['requests']['successful'] > 0:
            print(f"\nThroughput:")
            print(f"  Requests/second: {results['throughput']['requests_per_second']:.2f}")
            print(f"  Tokens/second: {results['throughput']['output_tokens_per_second']:.1f}")
            
            print(f"\nLatency:")
            if results['latency']['time_to_first_token_ms']['count'] > 0:
                print(f"  TTFT p50: {results['latency']['time_to_first_token_ms']['p50']:.0f} ms")
                print(f"  TTFT p90: {results['latency']['time_to_first_token_ms']['p90']:.0f} ms")
                print(f"  E2E p50: {results['latency']['end_to_end_latency_ms']['p50']/1000:.1f} s")
                print(f"  E2E p90: {results['latency']['end_to_end_latency_ms']['p90']/1000:.1f} s")
            
            print(f"\nWorkload:")
            print(f"  Avg tokens prefilled: {results['prefix_caching']['average_tokens_prefilled']:.0f}")
        
        # Save results
        os.makedirs("experiments/results/single_gpu_reasonable", exist_ok=True)
        with open("experiments/results/single_gpu_reasonable/summary.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Load TP=8 results for comparison
        try:
            with open("experiments/results/tp8_reasonable/summary.json", "r") as f:
                tp8_results = json.load(f)
        except:
            tp8_results = None
        
        print(f"\n{'='*60}")
        print("FAIR COMPARISON - Same Workload (1.5 req/s)")
        print(f"{'='*60}")
        
        print(f"\nSingle GPU Results:")
        print(f"  Success rate: {results['requests']['success_rate']:.1%}")
        if results['requests']['successful'] > 0:
            print(f"  Throughput: {results['throughput']['requests_per_second']:.2f} req/s")
            print(f"  TTFT p50: {results['latency']['time_to_first_token_ms']['p50']:.0f} ms")
            print(f"  E2E p50: {results['latency']['end_to_end_latency_ms']['p50']/1000:.1f} s")
        
        if tp8_results:
            print(f"\nTP=8 Results:")
            print(f"  Success rate: {tp8_results['requests']['success_rate']:.1%}")
            print(f"  Throughput: {tp8_results['throughput']['requests_per_second']:.2f} req/s")
            print(f"  TTFT p50: {tp8_results['latency']['time_to_first_token_ms']['p50']:.0f} ms")
            print(f"  E2E p50: {tp8_results['latency']['end_to_end_latency_ms']['p50']/1000:.1f} s")
            
            print(f"\n📊 Analysis:")
            if results['requests']['success_rate'] < tp8_results['requests']['success_rate']:
                improvement = (tp8_results['requests']['success_rate'] - results['requests']['success_rate']) * 100
                print(f"  TP=8 shows {improvement:.0f} percentage points higher success rate")
            
            if results['requests']['successful'] > 0 and tp8_results['latency']['time_to_first_token_ms']['count'] > 0:
                ttft_speedup = results['latency']['time_to_first_token_ms']['p50'] / tp8_results['latency']['time_to_first_token_ms']['p50']
                print(f"  TP=8 shows {ttft_speedup:.1f}x faster TTFT")
        
        print("\n💡 Key Insight:")
        if results['requests']['success_rate'] < 1.0:
            print("  Single GPU struggles even with reasonable load due to large prompt sizes")
        else:
            print("  With chunked prefill, both configurations can handle the workload")
            print("  The key difference is in latency and maximum throughput capacity")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()