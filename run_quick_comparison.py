#!/usr/bin/env python3
"""Run quick comparison of parallelization strategies with shorter simulations."""

import os
import sys
import json
import time
import yaml
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llmperforacle.orchestration.experiment_orchestrator import ExperimentOrchestrator

# Configurations to test
CONFIGS = [
    ("single_gpu", "configs/gh200_experiment_single.yaml"),
    ("tp8", "configs/gh200_experiment_tp8.yaml"),
    ("pp8", "configs/gh200_experiment_pp8.yaml"),
    ("dp8", "configs/gh200_experiment_dp8.yaml"),
    ("tp4pp2", "configs/gh200_experiment_tp4pp2.yaml"),
    ("tp2dp4", "configs/gh200_experiment_tp2dp4.yaml"),
    ("tp2pp2dp2", "configs/gh200_experiment_tp2pp2dp2.yaml"),
]

def run_quick_test(name, config_file):
    """Run a quick test with reduced simulation time."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    # Load and modify config for quick test
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Reduce simulation time to 60 seconds
    config['simulation']['max_simulation_time'] = 60
    config['workload']['total_duration'] = 60
    config['metrics_config']['warm_up_duration_s'] = 10  # Short warmup
    
    # Reduce workload intensity slightly to allow completion
    for profile in config['workload']['client_profiles']:
        if 'inter_arrival_time_dist_config' in profile:
            if profile['inter_arrival_time_dist_config']['type'] == 'Exponential':
                # Reduce rate by 50% to give more time between requests
                profile['inter_arrival_time_dist_config']['rate'] *= 0.5
    
    try:
        start_time = time.time()
        
        # Run simulation
        orchestrator = ExperimentOrchestrator(config)
        orchestrator.setup_simulation()
        
        # Print framework config
        framework = orchestrator.llm_framework_instances[0]
        print(f"Framework: {framework.__class__.__name__}")
        print(f"max_batched_tokens: {framework.max_batched_tokens_per_iteration}")
        if hasattr(framework, 'enable_chunked_prefill'):
            print(f"chunked_prefill: {framework.enable_chunked_prefill}")
        
        results = orchestrator.run()
        
        elapsed = time.time() - start_time
        
        print(f"\nCompleted in {elapsed:.1f}s")
        print(f"Success rate: {results['requests']['success_rate']:.1%}")
        print(f"Total requests: {results['requests']['total']}")
        print(f"Throughput: {results['throughput']['requests_per_second']:.2f} req/s")
        
        # Save results
        output_dir = f"experiments/results/quick_test_{name}"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        return {
            "name": name,
            "success": True,
            "results": results,
            "elapsed_time": elapsed
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "name": name,
            "success": False,
            "error": str(e),
            "elapsed_time": time.time() - start_time
        }

def main():
    """Run all quick tests and generate comparison."""
    print("Quick Parallelization Strategy Comparison")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNote: Using 60-second simulations with reduced workload for quick results")
    
    all_results = []
    
    for name, config_file in CONFIGS:
        result = run_quick_test(name, config_file)
        all_results.append(result)
        
        # Save intermediate results
        with open("experiments/results/quick_comparison_progress.json", "w") as f:
            json.dump(all_results, f, indent=2)
    
    # Generate summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    successful = [r for r in all_results if r.get("success", False)]
    
    if successful:
        # Create comparison table
        data = []
        for r in successful:
            res = r["results"]
            data.append({
                "Configuration": r["name"],
                "Success Rate": f"{res['requests']['success_rate']:.1%}",
                "Total Requests": res['requests']['total'],
                "Successful": res['requests']['successful'],
                "Failed": res['requests']['failed'],
                "Throughput (req/s)": f"{res['throughput']['requests_per_second']:.2f}",
                "TTFT p50 (ms)": f"{res['latency']['time_to_first_token_ms']['p50']:.1f}",
                "TTFT p99 (ms)": f"{res['latency']['time_to_first_token_ms']['p99']:.1f}",
                "E2E p50 (ms)": f"{res['latency']['end_to_end_latency_ms']['p50']:.1f}",
                "E2E p99 (ms)": f"{res['latency']['end_to_end_latency_ms']['p99']:.1f}",
            })
        
        df = pd.DataFrame(data)
        print("\nResults Table:")
        print(df.to_string(index=False))
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = f"experiments/results/quick_comparison_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"\nResults saved to: {csv_file}")
        
        # Find best configurations
        if len(df) > 0:
            # Convert percentage strings back to floats for comparison
            df['Success Rate Float'] = df['Success Rate'].str.rstrip('%').astype(float) / 100
            df['Throughput Float'] = df['Throughput (req/s)'].astype(float)
            
            best_success = df.loc[df['Success Rate Float'].idxmax()]
            best_throughput = df.loc[df['Throughput Float'].idxmax()]
            
            print(f"\nBest success rate: {best_success['Configuration']} ({best_success['Success Rate']})")
            print(f"Best throughput: {best_throughput['Configuration']} ({best_throughput['Throughput (req/s)']} req/s)")
            
            # Group by parallelization type
            print("\nAnalysis by parallelization type:")
            if 'tp8' in [r['name'] for r in successful]:
                print("- Tensor Parallel (TP): Good for handling large prompts with chunked prefill")
            if 'pp8' in [r['name'] for r in successful]:
                print("- Pipeline Parallel (PP): Good for memory distribution across stages")
            if 'dp8' in [r['name'] for r in successful]:
                print("- Data Parallel (DP): Good for high throughput with many instances")
            if any(r['name'] in ['tp4pp2', 'tp2dp4', 'tp2pp2dp2'] for r in successful):
                print("- Hybrid approaches: Balance benefits of multiple strategies")

if __name__ == "__main__":
    main()