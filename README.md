# FSDP vs PyTorch Baseline Training Benchmark using RTX 4090 GPU

## Overview
<img width="400" height="480" alt="image" src="https://github.com/user-attachments/assets/481da7e9-075c-45a5-a84e-5f979cda7283" />
<img width="400" height="480" alt="image" src="https://github.com/user-attachments/assets/f7fb5c03-4b56-4bfc-893d-140e3e51db93" />

# FSDP vs PyTorch Baseline Training Benchmark

## Overview

This project compares two training strategies for a GPT-style causal language model:

- **PyTorch baseline (single-process training loop)**
- **Fully Sharded Data Parallel (FSDP)**

The objective is to measure real system-level tradeoffs in:

- Step latency (ms/iteration)
- Training stability (loss convergence)
- Runtime overhead introduced by distributed abstractions

---

## Experimental Setup

### Model
- GPT-2–style causal language model
- Objective: next-token prediction (cross-entropy loss)

### Hardware
- Single NVIDIA GPU (Vast.ai instance)
- One process per experiment (no multi-GPU scaling in this run)

---

## Metrics Collected

Each step logs:

- **Step time (ms)** — wall-clock iteration latency
- **Training loss** — optimization objective
- **GPU utilization ID (debug field)**
- **Memory footprint (MB)**

---

## Results

# 1. Step Latency Comparison

### Key Observations (from plot)

#### Initial step overhead (cold start)
- PyTorch baseline: ~**313 ms**
- FSDP: ~**505 ms**

This spike reflects:
- CUDA kernel warmup
- Graph compilation / initialization
- FSDP wrapping + distributed context setup

---

#### Steady-state training latency (steps 1–50)

After warmup, both systems stabilize:

| Method | Typical Step Time |
|--------|------------------|
| PyTorch baseline | ~19–21 ms |
| FSDP | ~25–28 ms |

### Interpretation

- PyTorch baseline is consistently faster per step
- FSDP introduces ~20–35% overhead even without multi-GPU usage
- Overhead comes from:
  - Parameter sharding logic
  - Autograd hooks
  - Materialization/unsharding steps
  - Distributed scaffolding

---

# 2. Training Loss Comparison

### Key Observations (from plot)

Both methods show nearly identical convergence behavior:

#### Initial loss
- PyTorch: ~12.9
- FSDP: ~12.7

#### Final stabilized region (steps ~20–50)
- Both converge around: **10.9 – 11.1**

### Interpretation

- Loss curves overlap almost perfectly
- No degradation in optimization quality from FSDP
- Minor stochastic noise differences are expected due to:
  - Different execution ordering
  - Kernel scheduling variance
  - Non-deterministic GPU operations

---

## Key Findings

### 1. PyTorch Baseline is Faster in Single-GPU Setting

The baseline avoids distributed abstraction overhead:

- No parameter sharding
- No distributed autograd hooks
- No communication layers
- Minimal runtime indirection

➡ Result: lower latency per step

---

### 2. FSDP Has Real Overhead Even Without Scaling Benefit

Even on a single GPU:

- FSDP still constructs distributed execution graphs
- Wraps model parameters into sharded representations
- Executes synchronization logic that is effectively unnecessary in this setting

➡ Result: higher latency with no memory/scaling advantage in this setup

---

### 3. Initial Spike is Significant and Systemic

Both systems show a large first-step spike:

- PyTorch: ~313 ms
- FSDP: ~505 ms

This is caused by:
- CUDA context initialization
- Kernel caching
- Graph setup
- Memory allocator warmup
- FSDP wrapping overhead (extra cost for FSDP)

This is expected behavior in ML systems benchmarks and should NOT be averaged into steady-state performance.

---

### 4. Why FSDP Still Exists (Despite Being Slower Here)

FSDP is not designed for single-GPU optimization.

It provides:

| Property | PyTorch | FSDP |
|----------|--------|------|
| Single GPU speed | Faster | Slower |
| Multi-GPU scaling | Limited | Strong |
| Memory efficiency | Medium | High |
| Large model support | Limited | Excellent |

FSDP becomes essential when:
- Models exceed single GPU memory
- Training requires sharding optimizer states
- Scaling across multiple GPUs/nodes

---

## Conclusion

This benchmark demonstrates a fundamental systems tradeoff:

> **Abstraction for scalability introduces overhead in non-scaled environments**

In this experiment:

- PyTorch baseline = best for single-device efficiency
- FSDP = best for distributed and memory-constrained training

The identical loss curves confirm that:
> FSDP changes execution strategy, not learning dynamics.

---

## Future Work

To extend this benchmark meaningfully:

- Multi-GPU scaling efficiency curves (1 → 2 → 4 → 8 GPUs)
- Memory scaling vs model size
- Tokens/sec throughput benchmarking
- ZeRO-3 (DeepSpeed) comparison
- Mixed precision (FP16/BF16) impact
- Communication overhead profiling per layer

---
