#!/usr/bin/env python3
"""Test all parallelization configurations with heavy workload."""

import os
import sys
import json
import yaml
import time
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llmperforacle.orchestration.experiment_orchestrator import ExperimentOrchestrator

# Configurations to test
CONFIGS = [
    ("single_gpu", "configs/gh200_experiment_single.yaml", "Single GPU baseline"),
    ("tp8", "configs/gh200_experiment_tp8.yaml", "Tensor Parallelism (8 GPUs)"),
    ("pp8", "configs/gh200_experiment_pp8.yaml", "Pipeline Parallelism (8 stages)"),
    ("dp8", "configs/gh200_experiment_dp8.yaml", "Data Parallelism (8 replicas)"),
    ("tp4pp2", "configs/gh200_experiment_tp4pp2.yaml", "TP=4, PP=2 hybrid"),
    ("tp2dp4", "configs/gh200_experiment_tp2dp4.yaml", "TP=2, DP=4 hybrid"),
    ("tp2pp2dp2", "configs/gh200_experiment_tp2pp2dp2.yaml", "TP=2, PP=2, DP=2 hybrid"),
]

def ensure_chunked_prefill(config):
    """Ensure chunked prefill is enabled for frameworks that support it."""
    for framework in config.get('frameworks_to_test', []):
        if framework['type'] in ['VLLM', 'ParallelVLLM']:
            if 'config' not in framework:
                framework['config'] = {}
            framework['config']['enable_chunked_prefill'] = True
            framework['config']['prefill_chunk_size'] = 4096
            # Remove explicit max_num_batched_tokens to allow dynamic calculation
            framework['config'].pop('max_num_batched_tokens', None)
    return config

def run_heavy_workload_test(name, config_file, description):
    """Run a test with heavy workload configuration."""
    print(f"\n{'='*70}")
    print(f"Testing: {name} - {description}")
    print(f"Config: {config_file}")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}")
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # The configs already have the heavy workload with correct format
    # Just update simulation parameters for reasonable test time
    
    # Set simulation parameters for reasonable test time
    config['simulation']['max_simulation_time'] = 120  # 2 minutes
    config['workload']['total_duration'] = 120
    config['metrics_config']['warm_up_duration_s'] = 20  # 20 second warmup
    
    # Enable chunked prefill for all frameworks
    config = ensure_chunked_prefill(config)
    
    # Create output directory
    output_dir = f"experiments/results/heavy_{name}"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        start_time = time.time()
        
        # Run simulation
        orchestrator = ExperimentOrchestrator(config)
        orchestrator.setup_simulation()
        
        # Print framework configuration
        framework = orchestrator.llm_framework_instances[0]
        print(f"\nFramework configuration:")
        print(f"  Type: {framework.__class__.__name__}")
        if hasattr(framework, 'max_batched_tokens_per_iteration'):
            print(f"  max_batched_tokens: {framework.max_batched_tokens_per_iteration}")
        if hasattr(framework, 'enable_chunked_prefill'):
            print(f"  chunked_prefill: {framework.enable_chunked_prefill}")
            print(f"  chunk_size: {framework.prefill_chunk_size}")
        
        # Print parallelism info
        if hasattr(framework, 'tp_degree'):
            print(f"  TP degree: {framework.tp_degree}")
        if hasattr(framework, 'pp_stages'):
            print(f"  PP stages: {framework.pp_stages}")
        if hasattr(framework, 'data_parallel_size'):
            print(f"  DP size: {framework.data_parallel_size}")
        
        print("\nRunning simulation...")
        results = orchestrator.run()
        
        elapsed = time.time() - start_time
        
        # Print results
        print(f"\n✓ Completed in {elapsed:.1f}s")
        print(f"Results:")
        print(f"  Total requests: {results['requests']['total']}")
        print(f"  Successful: {results['requests']['successful']}")
        print(f"  Failed: {results['requests']['failed']}")
        print(f"  Success rate: {results['requests']['success_rate']:.1%}")
        print(f"  Throughput: {results['throughput']['requests_per_second']:.2f} req/s")
        
        if results['latency']['time_to_first_token_ms']['count'] > 0:
            print(f"  TTFT p50: {results['latency']['time_to_first_token_ms']['p50']:.1f} ms")
            print(f"  TTFT p99: {results['latency']['time_to_first_token_ms']['p99']:.1f} ms")
            print(f"  E2E p50: {results['latency']['end_to_end_latency_ms']['p50']:.1f} ms")
            print(f"  E2E p99: {results['latency']['end_to_end_latency_ms']['p99']:.1f} ms")
        
        # Save results
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # Also save configuration used
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f)
        
        return {
            "name": name,
            "description": description,
            "success": True,
            "results": results,
            "elapsed_time": elapsed
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Error after {elapsed:.1f}s: {str(e)}")
        return {
            "name": name,
            "description": description,
            "success": False,
            "error": str(e),
            "elapsed_time": elapsed
        }

def main():
    """Run all tests and generate comparison report."""
    print("Heavy Workload Testing on All Parallelization Strategies")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nUsing:")
    print("- Heavy workload with LogNormal(8.7, 0.17) prompt distribution")
    print("- ~6,000 tokens average prompt size")
    print("- Chunked prefill enabled (4096 tokens/chunk)")
    print("- 2-minute simulations with 20s warmup")
    
    all_results = []
    
    # Run all tests
    for name, config_file, description in CONFIGS:
        result = run_heavy_workload_test(name, config_file, description)
        all_results.append(result)
        
        # Save progress
        with open("experiments/results/heavy_workload_progress.json", "w") as f:
            json.dump(all_results, f, indent=2)
    
    # Generate final report
    print("\n" + "="*80)
    print("HEAVY WORKLOAD COMPARISON RESULTS")
    print("="*80)
    
    successful = [r for r in all_results if r.get("success", False)]
    failed = [r for r in all_results if not r.get("success", False)]
    
    if successful:
        # Create comparison table
        data = []
        for r in successful:
            res = r["results"]
            row = {
                "Configuration": r["name"],
                "Description": r["description"],
                "Success Rate": f"{res['requests']['success_rate']:.1%}",
                "Total Requests": res['requests']['total'],
                "Successful": res['requests']['successful'],
                "Failed": res['requests']['failed'],
                "Throughput (req/s)": res['throughput']['requests_per_second'],
                "Tokens/s": res['throughput']['output_tokens_per_second'],
            }
            
            # Add latency metrics if available
            if res['latency']['time_to_first_token_ms']['count'] > 0:
                row["TTFT p50 (ms)"] = res['latency']['time_to_first_token_ms']['p50']
                row["TTFT p99 (ms)"] = res['latency']['time_to_first_token_ms']['p99']
                row["E2E p50 (ms)"] = res['latency']['end_to_end_latency_ms']['p50']
                row["E2E p99 (ms)"] = res['latency']['end_to_end_latency_ms']['p99']
            else:
                row["TTFT p50 (ms)"] = "N/A"
                row["TTFT p99 (ms)"] = "N/A"
                row["E2E p50 (ms)"] = "N/A"
                row["E2E p99 (ms)"] = "N/A"
            
            data.append(row)
        
        df = pd.DataFrame(data)
        print("\nComparison Table:")
        print(df.to_string(index=False))
        
        # Save CSV
        csv_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = f"experiments/results/heavy_workload_comparison_{csv_timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"\nResults saved to: {csv_file}")
        
        # Identify best configurations
        if len(df) > 0:
            # Find best by success rate
            best_idx = df['Success Rate'].str.rstrip('%').astype(float).idxmax()
            best_success = df.iloc[best_idx]
            
            # Find best by throughput (among those with >50% success)
            high_success = df[df['Success Rate'].str.rstrip('%').astype(float) > 50]
            if len(high_success) > 0:
                best_throughput_idx = high_success['Throughput (req/s)'].idxmax()
                best_throughput = high_success.iloc[best_throughput_idx]
            else:
                best_throughput = best_success
            
            print(f"\n🏆 Best Configurations:")
            print(f"  Highest success rate: {best_success['Configuration']} ({best_success['Success Rate']})")
            print(f"  Best throughput (>50% success): {best_throughput['Configuration']} ({best_throughput['Throughput (req/s)']:.2f} req/s)")
    
    if failed:
        print(f"\n❌ Failed configurations: {len(failed)}")
        for r in failed:
            print(f"  - {r['name']}: {r.get('error', 'Unknown error')}")
    
    # Save full report
    report_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_parameters": {
            "workload": "heavy_workload",
            "simulation_time": "120s",
            "warmup_time": "20s",
            "chunked_prefill": True,
            "chunk_size": 4096
        },
        "results": all_results
    }
    
    report_file = f"experiments/results/heavy_workload_report_{report_timestamp}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nFull report saved to: {report_file}")
    
    # Print analysis
    print("\n📊 Analysis:")
    print("- Single GPU struggles with large prompts even with chunked prefill")
    print("- Tensor Parallelism (TP) distributes computation for large prompts")
    print("- Pipeline Parallelism (PP) distributes memory across stages")
    print("- Data Parallelism (DP) scales throughput but each instance limited by single GPU")
    print("- Hybrid approaches balance different benefits")

if __name__ == "__main__":
    main()