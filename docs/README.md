# LLMPerfOracle Documentation

This directory contains documentation for the LLMPerfOracle project.

## Contents

### Design Documents

The `design_documents/` directory contains the original design specifications:

1. **LLM Serving Virtual Testing Environment_.txt** - High-level design and architecture
2. **Document 1_ Core Simulation Engine.txt** - SimPy-based simulation core
3. **Document 2_ Parameterized Virtual Hardware Layer.txt** - Hardware modeling approach
4. **Document 3_ Configurable Workload Generator.txt** - Workload generation design
5. **Document 4_ Pluggable LLM Framework Module.txt** - Framework adapter pattern
6. **Document 5_ Metrics Collection and Reporting Module.txt** - Metrics system design
6. **Document 6_ Experiment Configuration and Orchestration.txt** - Experiment management

### Module Documentation

Each module has its own README.md file:

- `/src/llmperforacle/core/README.md` - Core simulation engine details
- `/src/llmperforacle/hardware/README.md` - Virtual hardware layer documentation
- `/src/llmperforacle/workload/README.md` - Workload generator documentation
- `/src/llmperforacle/frameworks/README.md` - Framework simulation guide
- `/src/llmperforacle/metrics/README.md` - Metrics collection documentation
- `/src/llmperforacle/orchestration/README.md` - Orchestration documentation

### Developer Guide

See `/CLAUDE.md` at the project root for:
- Development setup instructions
- Common patterns and pitfalls
- Debugging tips
- Extension guidelines

### User Guide

See `/README.md` at the project root for:
- Installation instructions
- Quick start guide
- Configuration examples
- API reference