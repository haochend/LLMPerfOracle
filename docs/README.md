# LLMPerfOracle Documentation

This directory contains comprehensive documentation for the LLMPerfOracle project, including design documents, implementation guides, and testing documentation.

## 📋 Table of Contents

1. [Design Documents](#design-documents)
2. [Implementation Guides](#implementation-guides)
3. [Feature Documentation](#feature-documentation)
4. [Testing Documentation](#testing-documentation)
5. [Module Documentation](#module-documentation)
6. [Developer Resources](#developer-resources)

## 📐 Design Documents

The `design_documents/` directory contains the original design specifications that guide the project architecture:

### Core Architecture
- **[LLM Serving Virtual Testing Environment](design_documents/LLM%20Serving%20Virtual%20Testing%20Environment_.txt)** - High-level design overview and project goals
- **[Document 1: Core Simulation Engine](design_documents/Document%201_%20Core%20Simulation%20Engine.txt)** - SimPy-based discrete event simulation core
- **[Document 2: Virtual Hardware Layer](design_documents/Document%202_%20Parameterized%20Virtual%20Hardware%20Layer.txt)** - Hardware modeling with cost models
- **[Document 3: Workload Generator](design_documents/Document%203_%20Configurable%20Workload%20Generator.txt)** - Request generation and workload patterns

### Framework & Systems
- **[Document 4: LLM Framework Module](design_documents/Document%204_%20Pluggable%20LLM%20Framework%20Module%20-%20Abstract%20Base%20%26%20vLLM%20Example.txt)** - Framework adapter pattern and vLLM implementation
- **[Document 5: Metrics Collection](design_documents/Document%205_%20Metrics%20Collection%20and%20Reporting%20Module.txt)** - Comprehensive metrics system
- **[Document 6: Orchestration](design_documents/Document%206_%20Experiment%20Configuration%20and%20Orchestration.txt)** - Experiment configuration and management

### Advanced Features
- **[Document 7: Multi-GPU Parallelism](design_documents/Document%207_%20Implementing%20Multi-GPU%20Parallelism%20(TP,%20PP,%20DP).txt)** - Tensor, Pipeline, and Data Parallelism
- **[Document 8: Prefix Caching](design_documents/Document%208_%20Simulating%20Prefix%20Caching.txt)** - KV cache reuse optimization (Phases 1-2 implemented, see [implementation status](prefix_caching_implementation_status.md))

## 🛠️ Implementation Guides

### General Guides
- **[Simulation Walkthrough](simulation_walkthrough.md)** - Step-by-step guide through a complete simulation
- **[Hardware Efficiency](hardware_efficiency.md)** - Understanding MFU/MBU and hardware utilization

### Parallelism Implementation
- **[Parallelism Testing Guide](parallelism_testing_guide.md)** - Comprehensive guide to testing multi-GPU configurations
  - Tensor Parallelism (TP) setup and testing
  - Pipeline Parallelism (PP) configuration
  - Data Parallelism (DP) load balancing
  - Combined parallelism strategies

## 🚀 Feature Documentation

### Prefix Caching (Phase 1 & 2)
Complete implementation of KV cache reuse optimization:

#### Phase 1 - Conversational Caching
1. **[Prefix Caching Implementation](prefix_caching_implementation.md)** - Technical implementation details
2. **[Prefix Caching Summary](prefix_caching_summary.md)** - High-level overview and design
3. **[Prefix Caching Test Results](prefix_caching_test_results.md)** - Comprehensive test coverage and results

#### Phase 2 - Cross-Request Caching  
4. **[Cross-Request Caching Implementation](cross_request_caching_implementation.md)** - Global prefix cache design and implementation

#### Supporting Enhancements
5. **[Workload Generator Enhancement](workload_generator_enhancement.md)** - Token accumulation for realistic conversations

#### Implementation Status
6. **[Prefix Caching Implementation Status](prefix_caching_implementation_status.md)** - Tracks which phases from Document 8 are implemented

Key achievements:
- ✅ 70%+ cache hit rate for conversational workloads
- ✅ 69% reduction in prefill computation
- ✅ 98% improvement in TTFT for cached requests
- ✅ Cross-request caching for common prefixes

## 🧪 Testing Documentation

- **[Test Fixes Summary](test_fixes_summary.md)** - Documentation of test issues and resolutions
  - Metrics calculation accuracy adjustments
  - Cross-request caching integration test fixes
  - All 27 prefix caching tests passing

## 📦 Module Documentation

Each core module has its own detailed README:

- **[Core Engine](../src/llmperforacle/core/README.md)** - Simulation environment and scheduling
- **[Virtual Hardware](../src/llmperforacle/hardware/README.md)** - Device modeling and resource management
- **[Workload Generator](../src/llmperforacle/workload/README.md)** - Request generation and patterns
- **[Framework Adapters](../src/llmperforacle/frameworks/README.md)** - LLM framework simulations
- **[Metrics System](../src/llmperforacle/metrics/README.md)** - Performance tracking and analysis
- **[Orchestration](../src/llmperforacle/orchestration/README.md)** - Experiment coordination

## 👩‍💻 Developer Resources

### Essential Guides
- **[CLAUDE.md](../CLAUDE.md)** - Comprehensive developer guide
  - Project structure and architecture
  - Common patterns and pitfalls
  - Debugging techniques
  - Extension guidelines
  - Performance optimization tips

### Quick References
- **[README.md](../README.md)** - Project overview and quickstart
  - Installation instructions
  - Basic usage examples
  - Configuration templates
  - CLI reference

## 🗂️ Documentation Organization

```
docs/
├── README.md                           # This file
├── design_documents/                   # Original design specifications
│   ├── Document 1-8_*.txt            # Core design documents
│   └── LLM Serving Virtual Testing Environment_.txt
├── simulation_walkthrough.md          # Step-by-step simulation guide
├── hardware_efficiency.md             # Hardware utilization guide
├── parallelism_testing_guide.md       # Multi-GPU testing guide
├── prefix_caching_implementation.md   # Phase 1 implementation
├── prefix_caching_summary.md          # Phase 1 overview
├── prefix_caching_test_results.md     # Test coverage and results
├── cross_request_caching_implementation.md  # Phase 2 implementation
├── workload_generator_enhancement.md  # Token accumulation feature
├── test_fixes_summary.md              # Test issue resolutions
└── prefix_caching_implementation_status.md  # Document 8 implementation status
```

## 📊 Key Metrics and Results

The implementation achieves:
- **Simulation Performance**: ~1000 requests/second simulation throughput
- **Hardware Accuracy**: Realistic MFU (50%) and MBU (80%) modeling
- **Parallelism Support**: TP, PP, DP with proper scaling behavior
- **Prefix Caching**: 70%+ hit rates, 98% TTFT improvement
- **Test Coverage**: 100+ tests across unit and integration suites

## 🔍 Finding Information

- **Architecture Questions** → Start with design documents
- **Implementation Details** → Check feature documentation
- **Code Examples** → See module READMEs and CLAUDE.md
- **Testing** → Review test documentation and guides
- **Troubleshooting** → Check CLAUDE.md pitfalls section

## 📝 Contributing Documentation

When adding new documentation:
1. Place design docs in `design_documents/`
2. Place implementation guides in the root `docs/` directory
3. Update this README.md with links to new documents
4. Follow the existing naming conventions
5. Include clear section headers and examples