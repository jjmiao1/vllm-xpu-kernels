"""
Unit test + benchmark for MiniMax QK Norm SYCL split kernels.

Usage (in docker, 2 XPU devices):
    # Correctness only:
    ZE_AFFINITY_MASK=4,5 torchrun --nproc_per_node=2 \
        tests/test_minimax_allreduce_rms_qk.py

    # Correctness + full benchmark:
    ZE_AFFINITY_MASK=4,5 torchrun --nproc_per_node=2 \
        tests/test_minimax_allreduce_rms_qk.py --bench

Compares three implementations:
  1. SYCL split kernels (minimax_qk_local_variance + allreduce + minimax_qk_rms_norm)
  2. Triton fallback (upstream vLLM: 2 Triton kernels + allreduce)
  3. Pure torch eager (split + pow + mean + cat + allreduce + rsqrt + mul)
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist

# Register the CCL backend for XPU distributed
try:
    import oneccl_bindings_for_pytorch  # noqa: F401
except ImportError:
    pass


# =============================================================================
# Implementation paths
# =============================================================================

def reference_eager(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_size: int,
    kv_size: int,
    tp_world: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch reference: matches _minimax_qk_norm_tp_eager."""
    q, k, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)
    orig_dtype = q.dtype
    q = q.to(torch.float32)
    k = k.to(torch.float32)

    q_var = q.pow(2).mean(dim=-1, keepdim=True)
    k_var = k.pow(2).mean(dim=-1, keepdim=True)

    qk_var = torch.cat([q_var, k_var], dim=-1)  # [num_tokens, 2]
    dist.all_reduce(qk_var, op=dist.ReduceOp.SUM)
    qk_var = qk_var / tp_world

    q_var, k_var = qk_var.chunk(2, dim=-1)
    q = q * torch.rsqrt(q_var + eps) * q_weight.float()
    k = k * torch.rsqrt(k_var + eps) * k_weight.float()
    return q.to(orig_dtype), k.to(orig_dtype)


def triton_fallback(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_size: int,
    kv_size: int,
    tp_rank: int,
    tp_world: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real Triton two-kernel path using the actual @triton.jit kernels.

    Imports the compiled Triton kernels from vLLM and calls them directly.
    Falls back to torch eager if Triton is not available.
    """
    num_tokens = qkv.shape[0]
    row_stride = qkv.stride(0)
    BLOCK = 1024
    grid = (num_tokens,)

    try:
        from vllm.model_executor.layers.minimax_rms_norm.rms_norm_tp import (
            _minimax_qk_var_kernel,
            _minimax_rms_apply_kernel,
        )

        # Kernel 1: compute variance
        qk_var = torch.empty(num_tokens, 2, dtype=torch.float32,
                             device=qkv.device)
        _minimax_qk_var_kernel[grid](
            qkv, qk_var, row_stride,
            q_size=q_size, kv_size=kv_size, BLOCK=BLOCK
        )

        # allreduce
        dist.all_reduce(qk_var, op=dist.ReduceOp.SUM)

        # Kernel 2: apply norm
        q_out = torch.empty(num_tokens, q_size, dtype=qkv.dtype,
                            device=qkv.device)
        k_out = torch.empty(num_tokens, kv_size, dtype=qkv.dtype,
                            device=qkv.device)
        _minimax_rms_apply_kernel[grid](
            qkv, qk_var, q_weight, k_weight, q_out, k_out,
            row_stride, q_size=q_size, kv_size=kv_size,
            tp_world=tp_world, eps=eps, BLOCK=BLOCK
        )
        return q_out, k_out

    except (ImportError, Exception):
        # Fallback to torch if Triton not available
        q, k, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q_var = q.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        k_var = k.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        qk_var = torch.cat([q_var, k_var], dim=-1)
        dist.all_reduce(qk_var, op=dist.ReduceOp.SUM)
        qk_var = qk_var / tp_world
        q_var, k_var = qk_var[:, :1], qk_var[:, 1:]
        q_f = q.to(torch.float32)
        k_f = k.to(torch.float32)
        q_out = (q_f * torch.rsqrt(q_var + eps) * q_weight.float()).to(qkv.dtype)
        k_out = (k_f * torch.rsqrt(k_var + eps) * k_weight.float()).to(qkv.dtype)
        return q_out, k_out


def sycl_split_path(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_size: int,
    kv_size: int,
    tp_world: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SYCL split kernels: var kernel + dist.all_reduce + norm kernel."""
    qk_var = torch.ops._C.minimax_qk_local_variance(qkv, q_size, kv_size)
    dist.all_reduce(qk_var, op=dist.ReduceOp.SUM)
    return torch.ops._C.minimax_qk_rms_norm(
        qkv, qk_var, q_weight, k_weight, q_size, kv_size, tp_world, eps
    )


# =============================================================================
# Benchmark helper
# =============================================================================

def bench_fn(fn, warmup=20, iters=200, sync_device=True):
    """Time a function (wall-clock). Returns average us per iteration.

    NOTE: This includes CPU dispatch overhead. In a pipelined model,
    CPU dispatch is hidden by GPU execution of other ops. Use bench_gpu_time()
    for the metric that reflects real model impact.
    """
    # Warmup
    for _ in range(warmup):
        fn()
    if sync_device:
        torch.xpu.synchronize()

    # Timed
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1_000_000  # us


def bench_gpu_time(fn, warmup=20, iters=100):
    """Measure GPU-side wall time using XPU events.

    Records timestamps on the GPU timeline (start before first call,
    end after last call). This gives total device-side elapsed time
    including kernels + inter-kernel gaps + communication, but excludes
    CPU-side overhead that would be hidden in a pipelined model.

    Returns average microseconds per call.
    """
    # Warmup
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        fn()
    end_event.record()
    torch.xpu.synchronize()

    # elapsed_time returns milliseconds
    elapsed_ms = start_event.elapsed_time(end_event)
    return elapsed_ms * 1000.0 / iters  # convert to μs per call


# L2 cache flush buffer (~210 MB, larger than PVC L2 of ~204 MB)
_L2_FLUSH_BUF = None


def _get_l2_flush_buf(device):
    """Lazily allocate a buffer larger than L2 cache for flushing."""
    global _L2_FLUSH_BUF
    if _L2_FLUSH_BUF is None or _L2_FLUSH_BUF.device != device:
        # 210 MB of float32 = 210*1024*1024/4 = 55,050,240 elements
        _L2_FLUSH_BUF = torch.zeros(55_050_240, dtype=torch.float32, device=device)
    return _L2_FLUSH_BUF


def bench_gpu_time_cold(fn, device, warmup=20, iters=100):
    """Measure GPU-side time with L2 cache flush between iterations.

    Uses paired XPU events per iteration so that only kernel execution
    time is measured (flush time is excluded). This simulates real model
    behavior where the input tensor is freshly produced by a preceding
    GEMM and NOT resident in L2.

    Returns average microseconds per call.
    """
    flush_buf = _get_l2_flush_buf(device)

    # Warmup (with flush to stabilize behavior)
    for _ in range(warmup):
        flush_buf.fill_(1.0)
        fn()
    torch.xpu.synchronize()

    # Timed: paired events per iteration (excludes flush time)
    starts = [torch.xpu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.xpu.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        flush_buf.fill_(1.0)  # evict working data from L2
        starts[i].record()
        fn()
        ends[i].record()
    torch.xpu.synchronize()

    total_us = sum(starts[i].elapsed_time(ends[i]) * 1000.0
                   for i in range(iters))
    return total_us / iters


# =============================================================================
# Main
# =============================================================================

def run_correctness(q_weight, k_weight, q_size, kv_size,
                    hidden_dim, dtype, device, rank, world_size, eps):
    """Run correctness tests across different token counts."""
    if rank == 0:
        print("=" * 70)
        print("CORRECTNESS TEST")
        print("=" * 70)

    token_counts = [1, 4, 32, 128, 512, 2048, 4096, 8192]
    all_passed = True

    for num_tokens in token_counts:
        torch.manual_seed(100 + rank)
        qkv = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)

        # Reference (pure torch)
        qkv_ref = qkv.clone()
        q_ref, k_ref = reference_eager(
            qkv_ref, q_weight, k_weight, q_size, kv_size, world_size, eps
        )

        # SYCL split kernels
        q_sycl, k_sycl = sycl_split_path(
            qkv, q_weight, k_weight, q_size, kv_size, world_size, eps
        )

        q_diff = (q_sycl - q_ref).abs().max().item()
        k_diff = (k_sycl - k_ref).abs().max().item()
        passed = q_diff < 0.02 and k_diff < 0.02  # BF16 tolerance

        if rank == 0:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] tokens={num_tokens:5d}: "
                  f"q_max_diff={q_diff:.6f}, k_max_diff={k_diff:.6f}")
        if not passed:
            all_passed = False

    if rank == 0:
        print()
        print("  RESULT:", "ALL PASSED ✓" if all_passed else "SOME FAILED ✗")
        print()
    return all_passed


def run_benchmark(q_weight, k_weight, q_size, kv_size,
                  hidden_dim, dtype, device, rank, world_size, eps):
    """Run performance benchmark: SYCL split vs Triton vs Eager.

    Three sections:
      1. Kernel-only GPU time with L2 COLD (paired events + flush) — matches real model
      2. Kernel-only GPU time with L2 WARM (original method) — for reference
      3. End-to-end wall-clock time (includes allreduce + CPU dispatch)
    """
    token_counts = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    iters = 100

    # =====================================================================
    # Part 1: Kernel-only GPU time — L2 COLD (realistic, matches profile)
    # =====================================================================

    # Try to load real Triton kernels
    triton_var_kernel = None
    triton_apply_kernel = None
    try:
        from vllm.model_executor.layers.minimax_rms_norm.rms_norm_tp import (
            _minimax_qk_var_kernel,
            _minimax_rms_apply_kernel,
        )
        triton_var_kernel = _minimax_qk_var_kernel
        triton_apply_kernel = _minimax_rms_apply_kernel
        has_triton = True
    except ImportError:
        has_triton = False

    if rank == 0:
        print("=" * 70)
        print("KERNEL-ONLY GPU TIME — L2 COLD (flush between iterations)")
        print(f"  Device: {device}, TP={world_size}, dtype={dtype}")
        print(f"  q_size={q_size}, kv_size={kv_size}, hidden={hidden_dim}")
        print(f"  Triton kernels: {'loaded' if has_triton else 'NOT AVAILABLE (using torch eager)'}")
        print("  NOTE: L2 flushed between iterations to simulate real model")
        print("=" * 70)
        print()

        # --- Var kernel comparison ---
        print("  [Variance kernel — COLD]  SYCL vs Triton")
        print(f"{'tokens':>8} | {'SYCL var':>10} | {'Triton var':>10} | {'speedup':>8}")
        print("-" * 50)

    BLOCK = 1024
    for num_tokens in token_counts:
        torch.manual_seed(200 + rank)
        qkv = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        row_stride = qkv.stride(0)
        grid = (num_tokens,)

        def triton_var_only():
            var = torch.empty(num_tokens, 2, dtype=torch.float32, device=device)
            if has_triton:
                triton_var_kernel[grid](
                    qkv, var, row_stride,
                    q_size=q_size, kv_size=kv_size, BLOCK=BLOCK)
            else:
                q, k, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)
                q_v = q.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
                k_v = k.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
                var = torch.cat([q_v, k_v], dim=-1)
            return var

        def sycl_var_only():
            return torch.ops._C.minimax_qk_local_variance(qkv, q_size, kv_size)

        gpu_sycl_var = bench_gpu_time_cold(sycl_var_only, device, warmup=30, iters=iters)
        gpu_triton_var = bench_gpu_time_cold(triton_var_only, device, warmup=30, iters=iters)

        if rank == 0:
            sp = gpu_triton_var / gpu_sycl_var if gpu_sycl_var > 0 else float('nan')
            print(f"{num_tokens:>8} | {gpu_sycl_var:>7.1f} us | {gpu_triton_var:>7.1f} us | {sp:>6.2f}x")

    if rank == 0:
        print()

        # --- Norm kernel comparison ---
        print("  [Norm kernel — COLD]  SYCL vs Triton")
        print(f"{'tokens':>8} | {'SYCL norm':>10} | {'Triton norm':>10} | {'speedup':>8}")
        print("-" * 50)

    for num_tokens in token_counts:
        torch.manual_seed(200 + rank)
        qkv = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        row_stride = qkv.stride(0)
        grid = (num_tokens,)

        # Pre-compute variance (same for both paths)
        qk_var = torch.ops._C.minimax_qk_local_variance(qkv, q_size, kv_size)
        dist.all_reduce(qk_var, op=dist.ReduceOp.SUM)

        def sycl_norm_only():
            return torch.ops._C.minimax_qk_rms_norm(
                qkv, qk_var, q_weight, k_weight,
                q_size, kv_size, world_size, eps)

        def triton_norm_only():
            q_out = torch.empty(num_tokens, q_size, dtype=dtype, device=device)
            k_out = torch.empty(num_tokens, kv_size, dtype=dtype, device=device)
            if has_triton:
                triton_apply_kernel[grid](
                    qkv, qk_var, q_weight, k_weight, q_out, k_out,
                    row_stride, q_size=q_size, kv_size=kv_size,
                    tp_world=world_size, eps=eps, BLOCK=BLOCK)
            else:
                q, k, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)
                q_var_t = qk_var[:, :1] / world_size
                k_var_t = qk_var[:, 1:]  / world_size
                q_out = (q.to(torch.float32) * torch.rsqrt(q_var_t + eps) * q_weight.float()).to(dtype)
                k_out = (k.to(torch.float32) * torch.rsqrt(k_var_t + eps) * k_weight.float()).to(dtype)
            return q_out, k_out

        gpu_sycl_norm = bench_gpu_time_cold(sycl_norm_only, device, warmup=30, iters=iters)
        gpu_triton_norm = bench_gpu_time_cold(triton_norm_only, device, warmup=30, iters=iters)

        if rank == 0:
            sp = gpu_triton_norm / gpu_sycl_norm if gpu_sycl_norm > 0 else float('nan')
            print(f"{num_tokens:>8} | {gpu_sycl_norm:>7.1f} us | {gpu_triton_norm:>7.1f} us | {sp:>6.2f}x")

    if rank == 0:
        print()

        # --- Combined var+norm (no allreduce) ---
        print("  [Var+Norm combined — COLD]  SYCL (2 kernels) vs Triton (2 kernels)")
        print(f"{'tokens':>8} | {'SYCL v+n':>10} | {'Triton v+n':>10} | {'speedup':>8}")
        print("-" * 50)

    for num_tokens in token_counts:
        torch.manual_seed(200 + rank)
        qkv = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
        row_stride = qkv.stride(0)
        grid = (num_tokens,)

        def sycl_both():
            v = torch.ops._C.minimax_qk_local_variance(qkv, q_size, kv_size)
            return torch.ops._C.minimax_qk_rms_norm(
                qkv, v, q_weight, k_weight,
                q_size, kv_size, world_size, eps)

        def triton_both():
            var = torch.empty(num_tokens, 2, dtype=torch.float32, device=device)
            q_out = torch.empty(num_tokens, q_size, dtype=dtype, device=device)
            k_out = torch.empty(num_tokens, kv_size, dtype=dtype, device=device)
            if has_triton:
                triton_var_kernel[grid](
                    qkv, var, row_stride,
                    q_size=q_size, kv_size=kv_size, BLOCK=BLOCK)
                triton_apply_kernel[grid](
                    qkv, var, q_weight, k_weight, q_out, k_out,
                    row_stride, q_size=q_size, kv_size=kv_size,
                    tp_world=world_size, eps=eps, BLOCK=BLOCK)
            else:
                q, k, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)
                q_f = q.to(torch.float32)
                k_f = k.to(torch.float32)
                q_v = q_f.pow(2).mean(dim=-1, keepdim=True)
                k_v = k_f.pow(2).mean(dim=-1, keepdim=True)
                q_out = (q_f * torch.rsqrt(q_v + eps) * q_weight.float()).to(dtype)
                k_out = (k_f * torch.rsqrt(k_v + eps) * k_weight.float()).to(dtype)
            return q_out, k_out

        gpu_sycl = bench_gpu_time_cold(sycl_both, device, warmup=30, iters=iters)
        gpu_triton = bench_gpu_time_cold(triton_both, device, warmup=30, iters=iters)

        if rank == 0:
            sp = gpu_triton / gpu_sycl if gpu_sycl > 0 else float('nan')
            print(f"{num_tokens:>8} | {gpu_sycl:>7.1f} us | {gpu_triton:>7.1f} us | {sp:>6.2f}x")

    # =====================================================================
    # Part 2: End-to-end wall-clock time (with allreduce)
    # =====================================================================
    if rank == 0:
        print()
        print("=" * 70)
        print("END-TO-END WALL-CLOCK TIME (with allreduce + CPU dispatch)")
        print("=" * 70)
        print()
        print(f"{'tokens':>8} | {'SYCL split':>12} | {'Triton 2-kernel':>15} | "
              f"{'Torch eager':>12} | {'vs Triton':>10} | {'vs Eager':>10}")
        print("-" * 80)

    for num_tokens in token_counts:
        torch.manual_seed(200 + rank)
        qkv = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)

        ms_sycl = bench_fn(
            lambda: sycl_split_path(
                qkv, q_weight, k_weight, q_size, kv_size, world_size, eps),
            warmup=20, iters=iters
        )
        ms_triton = bench_fn(
            lambda: triton_fallback(
                qkv, q_weight, k_weight, q_size, kv_size,
                rank, world_size, eps),
            warmup=20, iters=iters
        )
        ms_eager = bench_fn(
            lambda: reference_eager(
                qkv, q_weight, k_weight, q_size, kv_size, world_size, eps),
            warmup=20, iters=iters
        )

        if rank == 0:
            speedup_triton = ms_triton / ms_sycl if ms_sycl > 0 else float('nan')
            speedup_eager = ms_eager / ms_sycl if ms_sycl > 0 else float('nan')
            print(f"{num_tokens:>8} | {ms_sycl:>9.1f} us | {ms_triton:>12.1f} us | "
                  f"{ms_eager:>9.1f} us | {speedup_triton:>8.2f}x | {speedup_eager:>8.2f}x")

    if rank == 0:
        print()
        print("  speedup > 1 means SYCL is faster")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true",
                        help="Run full benchmark (default: correctness only)")
    parser.add_argument("--iters", type=int, default=200,
                        help="Benchmark iterations per token count")
    args = parser.parse_args()

    # Initialize distributed
    try:
        dist.init_process_group(backend="ccl")
    except (AssertionError, RuntimeError):
        # PyTorch 2.9+ may use "xccl" for XPU
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    device = torch.device(f"xpu:{local_rank}")
    torch.xpu.set_device(device)

    # Load extension
    try:
        import vllm_xpu_kernels._C  # noqa: F401
    except ImportError as e:
        if rank == 0:
            print(f"WARNING: Failed to import vllm_xpu_kernels._C: {e}")
        pass

    # Check split kernels are available
    var_op = getattr(torch.ops._C, "minimax_qk_local_variance", None)
    norm_op = getattr(torch.ops._C, "minimax_qk_rms_norm", None)
    if var_op is None or norm_op is None:
        if rank == 0:
            print("ERROR: SYCL split kernels not found!")
            print("  minimax_qk_local_variance:", var_op)
            print("  minimax_qk_rms_norm:", norm_op)
            print("Make sure vllm-xpu-kernels is built and installed.")
        dist.destroy_process_group()
        sys.exit(1)

    if rank == 0:
        print(f"\n  Backend: ccl, TP={world_size}, device={device}")
        print()

    # MiniMax-M2.7 config with TP=2:
    #   num_q_heads=32, num_kv_heads=8, head_dim=96
    #   q_size_per_tp = 32/2 * 96 = 1536
    #   kv_size_per_tp = 8/2 * 96 = 384
    q_size = 1536
    kv_size = 384
    hidden_dim = q_size + 2 * kv_size  # 2304
    eps = 1e-6
    dtype = torch.bfloat16

    torch.manual_seed(42)
    q_weight = torch.randn(q_size, dtype=dtype, device=device) * 0.1 + 1.0
    k_weight = torch.randn(kv_size, dtype=dtype, device=device) * 0.1 + 1.0

    # --- Correctness ---
    passed = run_correctness(
        q_weight, k_weight, q_size, kv_size,
        hidden_dim, dtype, device, rank, world_size, eps
    )

    if not passed:
        dist.destroy_process_group()
        sys.exit(1)

    # --- Benchmark ---
    if args.bench:
        run_benchmark(
            q_weight, k_weight, q_size, kv_size,
            hidden_dim, dtype, device, rank, world_size, eps
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
