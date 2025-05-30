# LLMPerfOracle Tests

This directory contains the test suite for LLMPerfOracle.

## Test Structure

### Unit Tests (`unit/`)
Individual component tests (to be implemented):
- Core simulation engine tests
- Hardware layer tests
- Workload generator tests
- Framework adapter tests
- Metrics collector tests

### Integration Tests (`integration/`)
End-to-end simulation tests:
- `test_basic_simulation.py` - Basic simulation functionality test

## Running Tests

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

# Run specific test file
pytest tests/test_basic_simulation.py
```

## Test Configuration

The basic simulation test uses a minimal configuration:
- 10-second simulation duration
- Single GPU with 16GB memory
- Constant token lengths (50 input, 100 output)
- 2 requests/second arrival rate
- vLLM framework with small batch sizes

## Adding New Tests

1. **Unit Tests**: Test individual components in isolation
   ```python
   def test_distribution_sampler():
       sampler = DistributionSampler()
       value = sampler.sample({"type": "Exponential", "rate": 5.0})
       assert value > 0
   ```

2. **Integration Tests**: Test complete simulation scenarios
   ```python
   def test_multi_framework_simulation():
       config = load_test_config("multi_framework.yaml")
       orchestrator = ExperimentOrchestrator(config)
       summary = orchestrator.run()
       assert_performance_metrics(summary)
   ```

## Test Utilities

Common test utilities should be placed in `tests/utils.py`:
- Configuration builders
- Assertion helpers
- Mock objects
- Test data generators