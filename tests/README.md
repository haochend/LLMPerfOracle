# LLMPerfOracle Tests

This directory contains the comprehensive test suite for LLMPerfOracle.

## Test Structure

### Unit Tests (`unit/`)
Focused tests for individual components:
- `test_parallelism.py` - Multi-GPU parallelism strategies (TP, PP, DP)
- `test_parallelism_edge_cases.py` - Edge cases and error conditions for parallelism
- `test_parallelism_performance.py` - Performance characteristics of parallel configurations
- `test_prefix_caching.py` - Conversational and cross-request caching functionality
- `test_prefix_caching_edge_cases.py` - Edge cases for prefix caching
- `test_cross_request_caching.py` - Global prefix cache implementation
- `test_config_validator.py` - Configuration validation system
- `test_performance_abstractions.py` - Level of Detail (LoD) simulation speedup

### Integration Tests (`integration/`)
End-to-end simulation tests:
- `test_basic_simulation.py` - Basic simulation functionality
- `test_parallel_simulation_quick.py` - Quick demos of parallelism benefits (TP, PP, DP)
- `test_parallel_simulation_improved.py` - Comprehensive parallelism testing
- `test_prefix_caching_integration.py` - End-to-end prefix caching scenarios
- `test_cross_request_caching_integration.py` - Cross-request cache behavior
- `test_lod_accuracy_quick.py` - LoD accuracy verification
- `test_dp_scaling.py` - Data parallelism scaling tests

### Regression Tests (`regression/`)
Comprehensive test suite ensuring system stability:
- `test_regression_suite.py` - Full regression test suite including:
  - Configuration validation
  - Memory validation
  - Parallelism strategies
  - LoD functionality
  - Feature testing (prefix caching, chunked prefill)

## Running Tests

### Quick Start
```bash
# Run the regression test suite (recommended)
python run_regression_tests.py

# Run quick regression tests only
python run_regression_tests.py --quick

# Run specific test categories
python run_regression_tests.py --specific unit
python run_regression_tests.py --specific integration
```

### Using pytest directly
```bash
# Set up environment
cd /Users/davidd/LLMPerfOracle
source venv/bin/activate
export PYTHONPATH=/Users/davidd/LLMPerfOracle/src

# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest --cov=llmperforacle tests/

# Run specific test categories
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
pytest tests/regression/    # Regression tests only

# Run specific test file
pytest tests/unit/test_parallelism.py -v

# Run tests matching a pattern
pytest -k "prefix_caching" -v
pytest -k "test_tp_speedup" -v
```

## Test Categories

### Performance Tests
Tests that verify performance characteristics:
- **Parallelism Tests**: Verify TP/PP/DP provide expected speedups
- **LoD Tests**: Verify simulation speedup with different detail levels
- **Scaling Tests**: Verify system scales with hardware resources

### Functional Tests
Tests that verify correct behavior:
- **Prefix Caching**: Verify cache hits, token savings, and TTFT improvements
- **Memory Validation**: Verify models are rejected when exceeding GPU memory
- **Configuration Validation**: Verify invalid configs are properly rejected

### Edge Case Tests
Tests for boundary conditions:
- **Parallelism Edge Cases**: Invalid GPU counts, duplicate IDs, etc.
- **Prefix Caching Edge Cases**: Empty sessions, zero-length prompts, etc.
- **Resource Exhaustion**: KV cache limits, network congestion, etc.

## Test Fixtures and Patterns

### Common Fixtures
Many tests use pytest fixtures for setup:
```python
@pytest.fixture
def quick_config():
    """Base configuration for quick tests."""
    return {
        "simulation": {"max_simulation_time": 20},
        "model_characteristics_db_path": "configs/model_params.json",
        # ... hardware and workload config
    }
```

### Mock Objects
Tests use mocks for isolated component testing:
```python
class MockVirtualHardware:
    """Mock hardware for unit tests."""
    def submit_computation_task(self, device_id, task):
        return self.simpy_env.timeout(0.001)
```

### Performance Assertions
Tests verify relative performance improvements:
```python
# Verify TP provides speedup
assert tp2_ttft < single_gpu_ttft * 0.7  # 30% improvement

# Verify DP scales throughput
assert dp2_throughput > single_throughput * 1.5  # 50% scaling
```

## Test Execution Times

- **Unit tests**: < 1 second each
- **Quick integration tests**: 10-30 seconds
- **Full integration tests**: 1-3 minutes
- **Regression suite (quick)**: ~2 minutes
- **Regression suite (full)**: ~5 minutes

## Continuous Integration

The test suite is designed for CI/CD:
1. Unit tests run on every commit
2. Quick integration tests run on PRs
3. Full test suite runs nightly
4. Regression tests gate releases