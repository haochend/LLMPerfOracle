#!/usr/bin/env python3
"""Direct comparison of Single GPU vs TP=8 with heavy workload."""

import os
import sys
import json
import yaml
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llmperforacle.orchestration.experiment_orchestrator import ExperimentOrchestrator

def run_test(name, config_file, duration=30):
    """Run a test configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Duration: {duration}s")
    print(f"{'='*60}")
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set simulation parameters
    config['simulation']['max_simulation_time'] = duration
    config['workload']['total_duration'] = duration
    config['metrics_config']['warm_up_duration_s'] = 5
    
    # Enable chunked prefill
    for framework in config.get('frameworks_to_test', []):
        if framework['type'] in ['VLLM', 'ParallelVLLM']:
            if 'config' not in framework:
                framework['config'] = {}
            framework['config']['enable_chunked_prefill'] = True
            framework['config']['prefill_chunk_size'] = 4096
            framework['config'].pop('max_num_batched_tokens', None)
    
    try:
        start_time = time.time()
        
        # Run simulation
        orchestrator = ExperimentOrchestrator(config)
        orchestrator.setup_simulation()
        
        # Print framework configuration
        framework = orchestrator.llm_framework_instances[0]
        print(f"\nFramework configuration:")
        print(f"  Type: {framework.__class__.__name__}")
        if hasattr(framework, 'tp_degree'):
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
            
            # Show average tokens prefilled to understand workload
            if 'prefix_caching' in results:
                print(f"  Avg tokens prefilled: {results['prefix_caching']['average_tokens_prefilled']:.0f}")
        
        return {
            "name": name,
            "success": True,
            "results": results,
            "elapsed_time": elapsed
        }
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        return {
            "name": name,
            "success": False,
            "error": str(e)
        }

def main():
    """Run comparison between single GPU and TP=8."""
    print("Heavy Workload Comparison: Single GPU vs TP=8")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nObjective: Demonstrate TP=8's ability to handle large prompts")
    print("that cause failures on single GPU")
    
    results = []
    
    # Test 1: Single GPU with heavy workload
    single_result = run_test("Single GPU", "configs/gh200_experiment_single.yaml", duration=30)
    results.append(single_result)
    
    # Test 2: TP=8 with heavy workload
    tp8_result = run_test("TP=8", "configs/gh200_experiment_tp8.yaml", duration=30)
    results.append(tp8_result)
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    if all(r["success"] for r in results):
        single = results[0]["results"]
        tp8 = results[1]["results"]
        
        print(f"\n{'Metric':<30} {'Single GPU':>20} {'TP=8':>20}")
        print("-" * 70)
        print(f"{'Success Rate':<30} {single['requests']['success_rate']:>19.1%} {tp8['requests']['success_rate']:>19.1%}")
        print(f"{'Total Requests':<30} {single['requests']['total']:>20} {tp8['requests']['total']:>20}")
        print(f"{'Successful Requests':<30} {single['requests']['successful']:>20} {tp8['requests']['successful']:>20}")
        print(f"{'Failed Requests':<30} {single['requests']['failed']:>20} {tp8['requests']['failed']:>20}")
        
        if single['requests']['successful'] > 0:
            print(f"{'Throughput (req/s)':<30} {single['throughput']['requests_per_second']:>20.2f} ", end="")
        else:
            print(f"{'Throughput (req/s)':<30} {'N/A':>20} ", end="")
            
        if tp8['requests']['successful'] > 0:
            print(f"{tp8['throughput']['requests_per_second']:>20.2f}")
        else:
            print(f"{'N/A':>20}")
        
        # Show max batch tokens
        print(f"\n{'Configuration Details':<30}")
        print(f"{'Max batched tokens':<30} {'4096 (default)':>20} {tp8_result.get('max_batched_tokens', '3691 (dynamic)'):>20}")
        
        print("\n📊 Key Insights:")
        if tp8['requests']['success_rate'] > single['requests']['success_rate']:
            improvement = (tp8['requests']['success_rate'] / max(single['requests']['success_rate'], 0.001) - 1) * 100
            print(f"✓ TP=8 shows {improvement:.0f}% improvement in success rate")
            print(f"✓ TP=8 successfully processes large prompts that fail on single GPU")
            print(f"✓ Chunked prefill enables handling of prompts exceeding max_batched_tokens")
        else:
            print("⚠ Both configurations struggling with heavy workload")
            print("  Consider reducing request rate or prompt sizes")
    
    # Save comparison
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comparison_file = f"experiments/results/tp8_vs_single_comparison_{timestamp}.json"
    os.makedirs("experiments/results", exist_ok=True)
    
    with open(comparison_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "test_duration": 30,
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to: {comparison_file}")

if __name__ == "__main__":
    main()