"""
Performance comparison tests for Level of Detail (LoD) simulation modes.

This test suite implements the test cases from Document 9 to demonstrate
and quantify the speedup achieved by using medium LoD abstractions.
"""

import json
import time
import os
from pathlib import Path
from typing import Dict, Any, Tuple
import pytest
import yaml

from llmperforacle.orchestration import ExperimentOrchestrator


class TestLoDPerformanceComparison:
    """Test suite for comparing high vs medium Level of Detail performance."""
    
    @pytest.fixture
    def base_configs_dir(self):
        """Get the base configs directory."""
        return Path(__file__).parent.parent.parent / "configs"
    
    @pytest.fixture
    def results_dir(self):
        """Get the results directory and ensure it exists."""
        results_dir = Path(__file__).parent.parent.parent / "experiments" / "results" / "lod_comparison"
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir
    
    def load_and_modify_config(self, config_path: Path, lod: str, suffix: str) -> Dict[str, Any]:
        """Load config and modify it for the specific LoD test."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update LoD setting
        config['simulation']['lod'] = lod
        
        # Update output paths to include LoD suffix
        metrics_config = config['metrics_config']
        if 'output_summary_json_path' in metrics_config:
            path = Path(metrics_config['output_summary_json_path'])
            new_path = path.parent / f"{path.stem}_{suffix}{path.suffix}"
            metrics_config['output_summary_json_path'] = str(new_path)
        
        if 'output_requests_csv_path' in metrics_config:
            path = Path(metrics_config['output_requests_csv_path'])
            new_path = path.parent / f"{path.stem}_{suffix}{path.suffix}"
            metrics_config['output_requests_csv_path'] = str(new_path)
        
        # Reduce simulation time for faster testing
        config['simulation']['max_simulation_time'] = 60  # 1 minute for tests
        config['workload']['total_duration'] = 60
        
        return config
    
    def run_simulation_and_measure(self, config: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Run simulation and measure wall-clock time."""
        start_time = time.time()
        
        orchestrator = ExperimentOrchestrator(config)
        summary_report = orchestrator.run()
        
        end_time = time.time()
        wall_clock_time = end_time - start_time
        
        return wall_clock_time, summary_report
    
    def compare_results(
        self, 
        high_lod_results: Tuple[float, Dict[str, Any]], 
        medium_lod_results: Tuple[float, Dict[str, Any]],
        test_name: str,
        results_dir: Path
    ) -> Dict[str, Any]:
        """Compare results between high and medium LoD runs."""
        high_time, high_report = high_lod_results
        medium_time, medium_report = medium_lod_results
        
        # Calculate speedup
        speedup_factor = high_time / medium_time
        
        # Extract key metrics
        def extract_metrics(report):
            # Handle the actual report structure
            latency_stats = report.get('latency', {})
            requests_stats = report.get('requests', {})
            throughput_stats = report.get('throughput', {})
            
            return {
                'avg_ttft_ms': latency_stats.get('time_to_first_token_ms', {}).get('mean', 0),
                'avg_tpot_ms': latency_stats.get('time_per_output_token_ms', {}).get('mean', 0),
                'avg_e2e_latency_ms': latency_stats.get('end_to_end_latency_ms', {}).get('mean', 0),
                'throughput_requests_per_s': throughput_stats.get('request_throughput_per_s', 0),
                'throughput_tokens_per_s': throughput_stats.get('tokens_throughput_per_s', 0),
                'completed_requests': requests_stats.get('successful', 0),
            }
        
        high_metrics = extract_metrics(high_report)
        medium_metrics = extract_metrics(medium_report)
        
        # Calculate metric differences (percentage)
        metric_diffs = {}
        for key in high_metrics:
            if high_metrics[key] > 0:
                diff_pct = abs(medium_metrics[key] - high_metrics[key]) / high_metrics[key] * 100
                metric_diffs[f'{key}_diff_pct'] = diff_pct
        
        comparison_result = {
            'test_name': test_name,
            'wall_clock_time': {
                'high_lod_seconds': high_time,
                'medium_lod_seconds': medium_time,
                'speedup_factor': speedup_factor,
            },
            'metrics': {
                'high_lod': high_metrics,
                'medium_lod': medium_metrics,
                'differences_pct': metric_diffs,
            },
            'event_counts': {
                'high_lod': high_report.get('simulation_stats', {}).get('total_events', 0),
                'medium_lod': medium_report.get('simulation_stats', {}).get('total_events', 0),
            }
        }
        
        # Save comparison results
        comparison_path = results_dir / f"{test_name}_comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison_result, f, indent=2)
        
        return comparison_result
    
    @pytest.mark.parametrize("config_file,test_name", [
        ("lod_test_single_gpu.yaml", "single_gpu_macro_ops"),
        ("lod_test_tp4.yaml", "tp4_analytical_collectives"),
        ("lod_test_tp2pp2.yaml", "tp2pp2_combined"),
    ])
    def test_lod_comparison(self, config_file, test_name, base_configs_dir, results_dir):
        """Test LoD comparison for different scenarios."""
        config_path = base_configs_dir / config_file
        
        # Skip if config doesn't exist
        if not config_path.exists():
            pytest.skip(f"Config file {config_file} not found")
        
        # Run with high LoD
        high_config = self.load_and_modify_config(config_path, "high", "high_lod")
        high_results = self.run_simulation_and_measure(high_config)
        
        # Run with medium LoD
        medium_config = self.load_and_modify_config(config_path, "medium", "medium_lod")
        medium_results = self.run_simulation_and_measure(medium_config)
        
        # Compare results
        comparison = self.compare_results(high_results, medium_results, test_name, results_dir)
        
        # Assertions
        speedup = comparison['wall_clock_time']['speedup_factor']
        assert speedup > 1.0, f"Medium LoD should be faster than high LoD, but speedup was {speedup}"
        
        # Check that key metrics are within acceptable bounds (e.g., 10% difference)
        max_acceptable_diff = 10.0  # 10% maximum difference
        metric_diffs = comparison['metrics']['differences_pct']
        
        for metric, diff in metric_diffs.items():
            assert diff <= max_acceptable_diff, (
                f"{metric} differs by {diff:.2f}% between high and medium LoD, "
                f"exceeding threshold of {max_acceptable_diff}%"
            )
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Test: {test_name}")
        print(f"Speedup Factor: {speedup:.2f}x")
        print(f"Wall Clock Time - High LoD: {comparison['wall_clock_time']['high_lod_seconds']:.2f}s")
        print(f"Wall Clock Time - Medium LoD: {comparison['wall_clock_time']['medium_lod_seconds']:.2f}s")
        print(f"Metric Differences:")
        for metric, diff in metric_diffs.items():
            print(f"  {metric}: {diff:.2f}%")
        print(f"{'='*60}\n")
    
    def test_generate_summary_report(self, results_dir):
        """Generate a summary report of all LoD comparison tests."""
        summary = {
            'test_results': [],
            'overall_stats': {
                'avg_speedup': 0,
                'min_speedup': float('inf'),
                'max_speedup': 0,
                'avg_metric_diff_pct': 0,
            }
        }
        
        # Find all comparison files
        comparison_files = list(results_dir.glob("*_comparison.json"))
        
        if not comparison_files:
            pytest.skip("No comparison results found to summarize")
        
        speedups = []
        all_metric_diffs = []
        
        for comp_file in comparison_files:
            with open(comp_file, 'r') as f:
                comp_data = json.load(f)
            
            summary['test_results'].append(comp_data)
            
            speedup = comp_data['wall_clock_time']['speedup_factor']
            speedups.append(speedup)
            
            metric_diffs = list(comp_data['metrics']['differences_pct'].values())
            all_metric_diffs.extend(metric_diffs)
        
        # Calculate overall stats
        if speedups:
            summary['overall_stats']['avg_speedup'] = sum(speedups) / len(speedups)
            summary['overall_stats']['min_speedup'] = min(speedups)
            summary['overall_stats']['max_speedup'] = max(speedups)
        
        if all_metric_diffs:
            summary['overall_stats']['avg_metric_diff_pct'] = sum(all_metric_diffs) / len(all_metric_diffs)
        
        # Save summary
        summary_path = results_dir / "lod_comparison_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print("LoD Comparison Summary")
        print(f"{'='*60}")
        print(f"Average Speedup: {summary['overall_stats']['avg_speedup']:.2f}x")
        print(f"Min Speedup: {summary['overall_stats']['min_speedup']:.2f}x")
        print(f"Max Speedup: {summary['overall_stats']['max_speedup']:.2f}x")
        print(f"Average Metric Difference: {summary['overall_stats']['avg_metric_diff_pct']:.2f}%")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])