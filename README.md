# Coruscant

> **Co-Designing GPU Kernel and Sparse Tensor Core to Advocate Unstructured Sparsity in Efficient LLM Inference**

### Prerequisites

- **Python**: 3.8+ (recommended: 3.12)
- **GPU**: Any NVIDIA GPU with compute capability over 8.0 is good for async copy instrucsics
- **CUDA**: Verfied with 12.3
- **Conda**


##  Part 1: Kernel Evaluation

### 1. **Build Kernels from Source**
```bash
cd ./kernel_sources
source init_env_vars.sh

# Build FlashLLM
cd ./flash_llm/build
make 

# Build Coruscant Kernel
cd ../../coruscant_kernel/build
make

# Build Coruscant STC
cd ../../coruscant_stc/build
make
```

### 2.  **Build Kernel Evaluation Script**
```bash
cd ./kernel_comparison
make
```

### 3. **Usage**
The performance comparison tool accepts the following parameters:

```bash
./spmm_perf <M> <K> <N> <Sparsity> <SplitK>
```

**Parameters:**
- `M, K, N`: SpMM matrix dimensions
- `Sparsity`: Sparsity percentage (0-100)
- `SplitK`: Split-K factor (default: 16)

#### **Example**
```bash
./spmm_perf 7168 7168 8 50 8
```

## Part 2: End-to-End Performance

Coming Soon:

- [ ] Model preparation: pruning and compression
- [ ] Coruscant porting to PyTorch
- [ ] End-2-end evaluation script



---

## Citation

If you use Coruscant in your research, please cite:

```bibtex
@article{coruscant2025,
  title={Coruscant: Co-Designing GPU Kernel and Sparse Tensor Core to Advocate Unstructured Sparsity in Efficient LLM Inference},
  author={Donghyeon Joo, Helya Hosseini, Ramyad Hadidi, Bahar Asgari},
  booktitle={In the Proceeding of 58th Annual IEEE/ACM International Symposium on Microarchitecture},
  year={2025}
}
```

