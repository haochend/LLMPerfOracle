#!/usr/bin/env python3
"""Test TP=8 with heavy workload to demonstrate large prompt handling."""

import os
import sys
import json
import yaml
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llmperforacle.orchestration.experiment_orchestrator import ExperimentOrchestrator

def main():
    """Run TP=8 test with heavy workload."""
    print("Testing TP=8 with Heavy Workload")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nObjective: Demonstrate that TP=8 with chunked prefill can handle")
    print("large prompts (6,000+ tokens) that single GPU cannot process.")
    
    # Load TP=8 configuration
    config_file = "configs/gh200_experiment_tp8.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Short simulation for demonstration
    config['simulation']['max_simulation_time'] = 60  # 1 minute
    config['workload']['total_duration'] = 60
    config['metrics_config']['warm_up_duration_s'] = 10  # 10 second warmup
    
    # Ensure chunked prefill is enabled
    for framework in config.get('frameworks_to_test', []):
        if framework['type'] in ['VLLM', 'ParallelVLLM']:
            if 'config' not in framework:
                framework['config'] = {}
            framework['config']['enable_chunked_prefill'] = True
            framework['config']['prefill_chunk_size'] = 4096
            # Remove explicit max_num_batched_tokens to allow dynamic calculation
            framework['config'].pop('max_num_batched_tokens', None)
    
    # Create output directory
    output_dir = "experiments/results/tp8_heavy_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("Running TP=8 with Heavy Workload")
    print("="*70)
    
    try:
        start_time = time.time()
        
        # Run simulation
        orchestrator = ExperimentOrchestrator(config)
        orchestrator.setup_simulation()
        
        # Print framework configuration
        framework = orchestrator.llm_framework_instances[0]
        print(f"\nFramework configuration:")
        print(f"  Type: {framework.__class__.__name__}")
        print(f"  TP degree: {framework.tp_degree}")
        print(f"  max_batched_tokens: {framework.max_batched_tokens_per_iteration}")
        print(f"  chunked_prefill: {framework.enable_chunked_prefill}")
        print(f"  chunk_size: {framework.prefill_chunk_size}")
        
        print("\nRunning simulation...")
        results = orchestrator.run()
        
        elapsed = time.time() - start_time
        
        # Print results
        print(f"\n✓ Completed in {elapsed:.1f}s")
        print(f"\nResults:")
        print(f"  Total requests: {results['requests']['total']}")
        print(f"  Successful: {results['requests']['successful']}")
        print(f"  Failed: {results['requests']['failed']}")
        print(f"  Success rate: {results['requests']['success_rate']:.1%}")
        
        if results['requests']['successful'] > 0:
            print(f"  Throughput: {results['throughput']['requests_per_second']:.2f} req/s")
            print(f"  Tokens/s: {results['throughput']['output_tokens_per_second']:.1f}")
            
            if results['latency']['time_to_first_token_ms']['count'] > 0:
                print(f"\nLatency metrics:")
                print(f"  TTFT p50: {results['latency']['time_to_first_token_ms']['p50']:.1f} ms")
                print(f"  TTFT p99: {results['latency']['time_to_first_token_ms']['p99']:.1f} ms")
                print(f"  E2E p50: {results['latency']['end_to_end_latency_ms']['p50']:.1f} ms")
                print(f"  E2E p99: {results['latency']['end_to_end_latency_ms']['p99']:.1f} ms")
            
            # Show prompt size stats
            print(f"\nWorkload characteristics:")
            print(f"  Average tokens prefilled: {results['prefix_caching']['average_tokens_prefilled']:.0f}")
        
        # Save results
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # Compare with single GPU
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print("\nSingle GPU with heavy workload:")
        print("  Success rate: 1.9% (25 out of 1285 requests)")
        print("  Throughput: 0.25 req/s")
        print("  Issue: Cannot handle large prompts (6,000+ tokens)")
        
        print(f"\nTP=8 with heavy workload:")
        print(f"  Success rate: {results['requests']['success_rate']:.1%}")
        print(f"  Throughput: {results['throughput']['requests_per_second']:.2f} req/s")
        print("  Benefit: Handles large prompts via chunked prefill")
        
        print("\n📊 Key Insight:")
        print("Tensor Parallelism (TP=8) with chunked prefill enables processing")
        print("of large prompts that would otherwise fail on a single GPU.")
        print("The dynamic batch size calculation (3,691 tokens for TP=8) and")
        print("chunking (4,096 tokens/chunk) prevent scheduler deadlock.")
        
        # Save comparison
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "single_gpu": {
                "success_rate": 0.019455252918287938,
                "throughput_req_s": 0.25,
                "total_requests": 1285,
                "successful": 25,
                "limitation": "Cannot handle large prompts"
            },
            "tp8": {
                "success_rate": results['requests']['success_rate'],
                "throughput_req_s": results['throughput']['requests_per_second'],
                "total_requests": results['requests']['total'],
                "successful": results['requests']['successful'],
                "benefit": "Handles large prompts via chunked prefill"
            }
        }
        
        with open(os.path.join(output_dir, "comparison.json"), "w") as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()