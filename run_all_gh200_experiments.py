#!/usr/bin/env python3
"""Run all GH200 experiments with different parallelization configurations."""

import os
import sys
import json
import time
import subprocess
import multiprocessing
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# List of experiments to run
EXPERIMENTS = [
    {"name": "single_gpu", "config": "configs/gh200_experiment_single.yaml", "description": "Single GPU baseline"},
    {"name": "tp8", "config": "configs/gh200_experiment_tp8.yaml", "description": "Tensor Parallelism (8 GPUs)"},
    {"name": "pp8", "config": "configs/gh200_experiment_pp8.yaml", "description": "Pipeline Parallelism (8 stages)"},
    {"name": "dp8", "config": "configs/gh200_experiment_dp8.yaml", "description": "Data Parallelism (8 replicas)"},
    {"name": "tp4pp2", "config": "configs/gh200_experiment_tp4pp2.yaml", "description": "TP=4, PP=2 hybrid"},
    {"name": "tp2dp4", "config": "configs/gh200_experiment_tp2dp4.yaml", "description": "TP=2, DP=4 hybrid"},
    {"name": "tp2pp2dp2", "config": "configs/gh200_experiment_tp2pp2dp2.yaml", "description": "TP=2, PP=2, DP=2 hybrid"},
]

def run_experiment(exp_info):
    """Run a single experiment and return results."""
    name = exp_info["name"]
    config = exp_info["config"]
    description = exp_info["description"]
    
    print(f"\n{'='*60}")
    print(f"Starting: {name} - {description}")
    print(f"Config: {config}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Create output directory
    output_dir = f"experiments/results/gh200_{name}_fixed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the experiment using subprocess for better control
    cmd = [
        sys.executable,
        "-m", "llmperforacle.cli",
        "run", config,
        "-l", "INFO"
    ]
    
    log_file = os.path.join(output_dir, "simulation.log")
    
    try:
        with open(log_file, "w") as log:
            # Run with a 15-minute timeout per experiment
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONPATH": os.path.join(os.path.dirname(__file__), 'src')}
            )
            
            # Wait for completion with timeout
            process.wait(timeout=900)  # 15 minutes
            
            return_code = process.returncode
            
        elapsed_time = time.time() - start_time
        
        # Check if results were generated
        summary_file = os.path.join(output_dir.replace("_fixed", ""), "summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                results = json.load(f)
            
            # Copy results to the new directory
            import shutil
            shutil.copy2(summary_file, os.path.join(output_dir, "summary.json"))
            
            requests_csv = os.path.join(output_dir.replace("_fixed", ""), "requests.csv")
            if os.path.exists(requests_csv):
                shutil.copy2(requests_csv, os.path.join(output_dir, "requests.csv"))
            
            success_rate = results["requests"]["success_rate"]
            total_requests = results["requests"]["total"]
            throughput = results["throughput"]["requests_per_second"]
            
            print(f"\n✓ {name} completed in {elapsed_time:.1f}s")
            print(f"  Success rate: {success_rate:.1%}")
            print(f"  Total requests: {total_requests}")
            print(f"  Throughput: {throughput:.2f} req/s")
            
            return {
                "name": name,
                "status": "completed",
                "elapsed_time": elapsed_time,
                "results": results,
                "return_code": return_code
            }
        else:
            print(f"\n✗ {name} failed - no results generated")
            return {
                "name": name,
                "status": "failed",
                "elapsed_time": elapsed_time,
                "error": "No results file generated",
                "return_code": return_code
            }
            
    except subprocess.TimeoutExpired:
        process.kill()
        elapsed_time = time.time() - start_time
        print(f"\n⚠ {name} timed out after {elapsed_time:.1f}s")
        
        # Try to get partial results
        summary_file = os.path.join(output_dir.replace("_fixed", ""), "summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                results = json.load(f)
            return {
                "name": name,
                "status": "timeout",
                "elapsed_time": elapsed_time,
                "results": results
            }
        else:
            return {
                "name": name,
                "status": "timeout",
                "elapsed_time": elapsed_time,
                "error": "Timeout with no results"
            }
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ {name} error: {str(e)}")
        return {
            "name": name,
            "status": "error",
            "elapsed_time": elapsed_time,
            "error": str(e)
        }

def main():
    """Run all experiments and generate summary report."""
    print("GH200 Parallelization Experiments")
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Running {len(EXPERIMENTS)} experiments...")
    
    # Check if we should run in parallel
    parallel = "--parallel" in sys.argv
    
    if parallel:
        print("\nRunning experiments in parallel (may impact results accuracy)...")
        num_workers = min(4, multiprocessing.cpu_count() // 2)  # Use half the CPUs
        print(f"Using {num_workers} workers")
        
        with multiprocessing.Pool(num_workers) as pool:
            all_results = pool.map(run_experiment, EXPERIMENTS)
    else:
        print("\nRunning experiments sequentially...")
        all_results = []
        for exp in EXPERIMENTS:
            result = run_experiment(exp)
            all_results.append(result)
    
    # Generate summary report
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    completed = [r for r in all_results if r["status"] == "completed"]
    failed = [r for r in all_results if r["status"] in ["failed", "error", "timeout"]]
    
    print(f"\nCompleted: {len(completed)}/{len(EXPERIMENTS)}")
    
    if completed:
        print("\nResults Summary:")
        print(f"{'Config':<15} {'Success Rate':<12} {'Throughput':<15} {'Requests':<10} {'Time':<10}")
        print("-"*70)
        
        for result in completed:
            res = result["results"]
            print(f"{result['name']:<15} "
                  f"{res['requests']['success_rate']:<12.1%} "
                  f"{res['throughput']['requests_per_second']:<15.2f} "
                  f"{res['requests']['total']:<10} "
                  f"{result['elapsed_time']:<10.1f}s")
    
    if failed:
        print(f"\nFailed experiments: {len(failed)}")
        for result in failed:
            print(f"  - {result['name']}: {result.get('error', result['status'])}")
    
    # Save full results
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": len(EXPERIMENTS),
        "completed": len(completed),
        "failed": len(failed),
        "results": all_results
    }
    
    report_file = f"experiments/results/gh200_parallel_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nFull report saved to: {report_file}")
    
    # Generate comparison CSV for easy analysis
    if completed:
        import pandas as pd
        
        comparison_data = []
        for result in completed:
            res = result["results"]
            comparison_data.append({
                "configuration": result["name"],
                "success_rate": res["requests"]["success_rate"],
                "total_requests": res["requests"]["total"],
                "successful_requests": res["requests"]["successful"],
                "failed_requests": res["requests"]["failed"],
                "throughput_req_s": res["throughput"]["requests_per_second"],
                "throughput_tokens_s": res["throughput"]["output_tokens_per_second"],
                "ttft_p50_ms": res["latency"]["time_to_first_token_ms"]["p50"],
                "ttft_p99_ms": res["latency"]["time_to_first_token_ms"]["p99"],
                "e2e_p50_ms": res["latency"]["end_to_end_latency_ms"]["p50"],
                "e2e_p99_ms": res["latency"]["end_to_end_latency_ms"]["p99"],
                "simulation_time_s": result["elapsed_time"]
            })
        
        df = pd.DataFrame(comparison_data)
        csv_file = report_file.replace(".json", ".csv")
        df.to_csv(csv_file, index=False)
        print(f"Comparison CSV saved to: {csv_file}")
        
        # Print best configuration
        best_throughput = df.loc[df["throughput_req_s"].idxmax()]
        best_success = df.loc[df["success_rate"].idxmax()]
        
        print(f"\nBest throughput: {best_throughput['configuration']} ({best_throughput['throughput_req_s']:.2f} req/s)")
        print(f"Best success rate: {best_success['configuration']} ({best_success['success_rate']:.1%})")

if __name__ == "__main__":
    # Activate virtual environment if needed
    venv_path = os.path.join(os.path.dirname(__file__), "venv", "bin", "activate")
    if os.path.exists(venv_path):
        activate_cmd = f"source {venv_path}"
        print(f"Note: Make sure virtual environment is activated: {activate_cmd}")
    
    main()