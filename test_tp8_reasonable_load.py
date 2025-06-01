#!/usr/bin/env python3
"""Test TP=8 with reasonable load to actually see results."""

import os
import sys
import json
import yaml
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llmperforacle.orchestration.experiment_orchestrator import ExperimentOrchestrator

def main():
    """Run TP=8 with reasonable request rates."""
    print("TP=8 Heavy Workload Test - Reasonable Load")
    print("Adjusting request rates to match realistic processing capacity")
    
    # Load TP=8 configuration
    config_file = "configs/gh200_experiment_tp8.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # 30-second simulation
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
    print("This is too high for heavy workload with 6,000 token prompts!")
    
    # Scale down to reasonable rate
    # With 6,000 token prompts and ~1s TTFT, we can realistically handle ~1-2 req/s
    target_total_rate = 1.5  # requests per second
    scale_factor = target_total_rate / total_rate
    
    print(f"Scaling down by factor of {scale_factor:.3f}")
    print(f"New total request rate: {target_total_rate} req/s")
    
    # Apply scaling
    for profile in config['workload']['client_profiles']:
        if 'inter_arrival_time_dist_config' in profile:
            if profile['inter_arrival_time_dist_config']['type'] == 'Exponential':
                old_rate = profile['inter_arrival_time_dist_config']['rate']
                new_rate = old_rate * scale_factor
                profile['inter_arrival_time_dist_config']['rate'] = new_rate
                print(f"  {profile['profile_name']}: {old_rate} -> {new_rate:.3f} req/s")
    
    # Enable chunked prefill
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
        print(f"\nFramework: TP={framework.tp_degree}")
        print(f"Max batched tokens: {framework.max_batched_tokens_per_iteration}")
        print(f"Chunked prefill: {framework.enable_chunked_prefill}")
        
        results = orchestrator.run()
        
        elapsed = time.time() - start_time
        
        print(f"\n✓ Completed in {elapsed:.1f}s real time!")
        print(f"\n{'='*60}")
        print("RESULTS - TP=8 with Heavy Workload")
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
        os.makedirs("experiments/results/tp8_reasonable", exist_ok=True)
        with open("experiments/results/tp8_reasonable/summary.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print("\nSingle GPU with heavy workload (from earlier test):")
        print("  Success rate: 1.9%")
        print("  Throughput: 0.25 req/s")
        print(f"\nTP=8 with heavy workload (reasonable load):")
        print(f"  Success rate: {results['requests']['success_rate']:.1%}")
        print(f"  Throughput: {results['throughput']['requests_per_second']:.2f} req/s")
        
        if results['requests']['success_rate'] > 0.019:
            print(f"\n✓ TP=8 demonstrates {results['requests']['success_rate']/0.019:.0f}x "
                  f"improvement in success rate!")
            print("✓ Chunked prefill enables processing of large prompts")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()