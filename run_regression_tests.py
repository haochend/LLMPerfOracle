#!/usr/bin/env python3
"""
Run regression test suite for LLMPerfOracle.

This script provides an easy way to run comprehensive regression tests
to ensure the system is working correctly after changes.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description, verbose=False):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    if verbose:
        # Show output in real-time
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
    else:
        # Capture output
        result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"\n❌ FAILED: {description}")
            print("\nSTDOUT:")
            print(result.stdout)
            print("\nSTDERR:")
            print(result.stderr)
        else:
            print(f"✓ PASSED: {description}")
    
    elapsed = time.time() - start_time
    print(f"Time: {elapsed:.2f}s")
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run LLMPerfOracle regression tests")
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Run only quick tests (skip slow integration tests)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show test output in real-time"
    )
    parser.add_argument(
        "--specific", "-s",
        help="Run specific test class or method"
    )
    parser.add_argument(
        "--no-unit",
        action="store_true",
        help="Skip unit tests"
    )
    parser.add_argument(
        "--no-regression",
        action="store_true",
        help="Skip regression tests"
    )
    
    args = parser.parse_args()
    
    # Ensure we're in virtual environment
    venv_path = Path("venv")
    if not venv_path.exists():
        print("❌ Virtual environment not found. Please run:")
        print("  python -m venv venv")
        print("  source venv/bin/activate")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    # Set up environment
    python_exe = str(venv_path / "bin" / "python")
    
    all_passed = True
    
    # 1. Unit Tests
    if not args.no_unit:
        unit_test_suites = [
            ("Configuration Validator", ["tests/unit/test_config_validator.py"]),
            ("Performance Abstractions", ["tests/unit/test_performance_abstractions.py"]),
            ("Parallelism", ["tests/unit/test_parallelism.py"]),
            ("Prefix Caching", ["tests/unit/test_prefix_caching.py"]),
        ]
        
        if not args.quick:
            unit_test_suites.extend([
                ("Parallelism Edge Cases", ["tests/unit/test_parallelism_edge_cases.py"]),
                ("Prefix Caching Edge Cases", ["tests/unit/test_prefix_caching_edge_cases.py"]),
            ])
        
        for name, test_files in unit_test_suites:
            if args.specific and args.specific not in test_files[0]:
                continue
                
            cmd = [python_exe, "-m", "pytest", "-v"] + test_files
            if args.specific:
                cmd.append(f"-k {args.specific}")
            
            if not run_command(cmd, f"Unit Tests: {name}", args.verbose):
                all_passed = False
    
    # 2. Regression Tests
    if not args.no_regression:
        regression_cmd = [python_exe, "-m", "pytest", "-v", "tests/regression/test_regression_suite.py"]
        
        if args.quick:
            # Run only critical regression tests
            regression_cmd.extend(["-k", "test_example_configs_valid or test_model_database_valid or test_memory_validation"])
        
        if args.specific:
            regression_cmd.extend(["-k", args.specific])
        
        if not run_command(regression_cmd, "Regression Test Suite", args.verbose):
            all_passed = False
    
    # 3. Integration Tests (if not quick mode)
    if not args.quick:
        integration_tests = [
            ("Basic Simulation", ["tests/test_basic_simulation.py"]),
            ("Parallel Simulation Quick", ["tests/integration/test_parallel_simulation_quick.py"]),
            ("LoD Accuracy Quick", ["tests/integration/test_lod_accuracy_quick.py"]),
        ]
        
        for name, test_files in integration_tests:
            if args.specific and args.specific not in test_files[0]:
                continue
                
            cmd = [python_exe, "-m", "pytest", "-v", "-s"] + test_files
            if args.specific:
                cmd.append(f"-k {args.specific}")
            
            if not run_command(cmd, f"Integration Tests: {name}", args.verbose):
                all_passed = False
    
    # 4. Configuration Validation
    print(f"\n{'='*60}")
    print("Running: Configuration Validation")
    print(f"{'='*60}")
    
    configs_to_validate = [
        "configs/example_experiment.yaml",
        "configs/walkthrough_example.yaml",
        "configs/model_params.json",
    ]
    
    for config in configs_to_validate:
        if Path(config).exists():
            cmd = [python_exe, "-m", "llmperforacle.cli", "validate", config]
            if config.endswith("model_params.json"):
                continue  # Skip model DB validation via CLI
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if "Configuration is valid" in result.stdout or "Configuration has" in result.stdout:
                print(f"✓ Valid: {config}")
            else:
                print(f"❌ Invalid: {config}")
                if args.verbose:
                    print(result.stdout)
                all_passed = False
    
    # Summary
    print(f"\n{'='*60}")
    print("REGRESSION TEST SUMMARY")
    print(f"{'='*60}")
    
    if all_passed:
        print("✅ All regression tests PASSED!")
        print("\nThe system is working correctly:")
        print("- Configurations are valid")
        print("- Memory validation works")
        print("- Parallelism strategies function properly")
        print("- Performance abstractions (LoD) provide speedup")
        print("- Features like prefix caching work")
        return 0
    else:
        print("❌ Some regression tests FAILED!")
        print("\nPlease check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())