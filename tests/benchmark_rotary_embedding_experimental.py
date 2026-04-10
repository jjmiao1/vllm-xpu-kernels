# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import torch
import vllm_xpu_kernels._C  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.ops.rotary_embedding_op import RotaryEmbedding


# Mirrors the rotary_embedding workloads in xpu-perf/micro_perf/workloads/
# llm/single_test_ops/pre_fa_ops.json so the first printed results reflect the
# customer-visible shapes before the synthetic sweep.
CLIENT_PRIORITY_CASES = [
    {
        "attn_mode": "prefill",
        "dtype": torch.bfloat16,
        "batch_size": 1,
        "seq_len": 10240,
        "num_heads": 80,
        "num_kv_heads": 8,
        "head_size": 128,
        "rotary_dim": 128,
        "use_key": True,
        "head_stride_is_contiguous": True,
    },
    {
        "attn_mode": "prefill",
        "dtype": torch.bfloat16,
        "batch_size": 1,
        "seq_len": 5120,
        "num_heads": 80,
        "num_kv_heads": 8,
        "head_size": 128,
        "rotary_dim": 128,
        "use_key": True,
        "head_stride_is_contiguous": True,
    },
    {
        "attn_mode": "prefill",
        "dtype": torch.bfloat16,
        "batch_size": 1,
        "seq_len": 32768,
        "num_heads": 80,
        "num_kv_heads": 8,
        "head_size": 128,
        "rotary_dim": 128,
        "use_key": True,
        "head_stride_is_contiguous": True,
    },
    {
        "attn_mode": "decode",
        "dtype": torch.bfloat16,
        "batch_size": 16,
        "seq_len": 1,
        "num_heads": 80,
        "num_kv_heads": 8,
        "head_size": 128,
        "rotary_dim": 128,
        "use_key": True,
        "head_stride_is_contiguous": True,
    },
    {
        "attn_mode": "decode",
        "dtype": torch.bfloat16,
        "batch_size": 16,
        "seq_len": 4,
        "num_heads": 80,
        "num_kv_heads": 8,
        "head_size": 128,
        "rotary_dim": 128,
        "use_key": True,
        "head_stride_is_contiguous": True,
    },
]


# Mirrors the rotary_embedding workloads in xpu-perf/micro_perf/workloads/
# llm/igpu_ops/attn_ops.json. The current xpu-perf vllm_xpu_kernels backend
# builds a cache with width=rope_dim and calls the op directly, so these cases
# are benchmarked with the same effective contract here.
ATTN_OPS_PRIORITY_CASES = [
    {
        "attn_mode": "prefill",
        "dtype": torch.bfloat16,
        "batch_size": 1,
        "seq_len": 4096,
        "cache_len": 100,
        "num_heads": 96,
        "num_kv_heads": 8,
        "head_size": 128,
        "rope_offset": 0,
        "rotary_dim": 128,
        "use_key": True,
        "head_stride_is_contiguous": True,
    },
    {
        "attn_mode": "prefill",
        "dtype": torch.bfloat16,
        "batch_size": 1,
        "seq_len": 4096,
        "cache_len": 100,
        "num_heads": 96,
        "num_kv_heads": 8,
        "head_size": 128,
        "rope_offset": 80,
        "rotary_dim": 48,
        "use_key": True,
        "head_stride_is_contiguous": True,
    },
    {
        "attn_mode": "prefill",
        "dtype": torch.bfloat16,
        "batch_size": 1,
        "seq_len": 4096,
        "cache_len": 100,
        "num_heads": 96,
        "num_kv_heads": 8,
        "head_size": 128,
        "rope_offset": 80,
        "rotary_dim": 16,
        "use_key": True,
        "head_stride_is_contiguous": True,
    },
    {
        "attn_mode": "decode",
        "dtype": torch.bfloat16,
        "batch_size": 10,
        "seq_len": 4,
        "cache_len": 105,
        "num_heads": 96,
        "num_kv_heads": 8,
        "head_size": 128,
        "rope_offset": 0,
        "rotary_dim": 128,
        "use_key": True,
        "head_stride_is_contiguous": True,
    },
    {
        "attn_mode": "decode",
        "dtype": torch.bfloat16,
        "batch_size": 20,
        "seq_len": 4,
        "cache_len": 105,
        "num_heads": 96,
        "num_kv_heads": 8,
        "head_size": 128,
        "rope_offset": 0,
        "rotary_dim": 128,
        "use_key": True,
        "head_stride_is_contiguous": True,
    },
    {
        "attn_mode": "decode",
        "dtype": torch.bfloat16,
        "batch_size": 10,
        "seq_len": 4,
        "cache_len": 105,
        "num_heads": 96,
        "num_kv_heads": 8,
        "head_size": 128,
        "rope_offset": 80,
        "rotary_dim": 48,
        "use_key": True,
        "head_stride_is_contiguous": True,
    },
    {
        "attn_mode": "decode",
        "dtype": torch.bfloat16,
        "batch_size": 20,
        "seq_len": 4,
        "cache_len": 105,
        "num_heads": 96,
        "num_kv_heads": 8,
        "head_size": 128,
        "rope_offset": 80,
        "rotary_dim": 48,
        "use_key": True,
        "head_stride_is_contiguous": True,
    },
    {
        "attn_mode": "decode",
        "dtype": torch.bfloat16,
        "batch_size": 10,
        "seq_len": 4,
        "cache_len": 105,
        "num_heads": 96,
        "num_kv_heads": 8,
        "head_size": 128,
        "rope_offset": 80,
        "rotary_dim": 16,
        "use_key": True,
        "head_stride_is_contiguous": True,
    },
    {
        "attn_mode": "decode",
        "dtype": torch.bfloat16,
        "batch_size": 20,
        "seq_len": 4,
        "cache_len": 105,
        "num_heads": 96,
        "num_kv_heads": 8,
        "head_size": 128,
        "rope_offset": 80,
        "rotary_dim": 16,
        "use_key": True,
        "head_stride_is_contiguous": True,
    },
    {
        "attn_mode": "prefill_session_cache",
        "dtype": torch.bfloat16,
        "batch_size": 10,
        "seq_len": 4096,
        "cache_len": 100,
        "num_heads": 96,
        "num_kv_heads": 8,
        "head_size": 128,
        "rope_offset": 0,
        "rotary_dim": 128,
        "use_key": True,
        "head_stride_is_contiguous": True,
    },
    {
        "attn_mode": "prefill_session_cache",
        "dtype": torch.bfloat16,
        "batch_size": 10,
        "seq_len": 4096,
        "cache_len": 100,
        "num_heads": 96,
        "num_kv_heads": 8,
        "head_size": 128,
        "rope_offset": 80,
        "rotary_dim": 48,
        "use_key": True,
        "head_stride_is_contiguous": True,
    },
    {
        "attn_mode": "prefill_session_cache",
        "dtype": torch.bfloat16,
        "batch_size": 10,
        "seq_len": 4096,
        "cache_len": 100,
        "num_heads": 96,
        "num_kv_heads": 8,
        "head_size": 128,
        "rope_offset": 80,
        "rotary_dim": 16,
        "use_key": True,
        "head_stride_is_contiguous": True,
    },
]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_optional_int(value: str) -> int | None:
    stripped = value.strip().lower()
    if not stripped or stripped == "all":
        return None
    return int(stripped)


def parse_bool_list(value: str) -> list[bool]:
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    result = []
    for item in value.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(f"Unsupported boolean value: {item}")
        result.append(mapping[key])
    return result


def parse_dtype_list(value: str) -> list[torch.dtype]:
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    result = []
    for item in value.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(f"Unsupported dtype: {item}")
        result.append(mapping[key])
    return result


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def benchmark(fn, warmup: int, iters: int, timer: str) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    if timer == "host":
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.xpu.synchronize()
        return (time.perf_counter() - start) / iters * 1e6

    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    total_time_ms = 0.0
    for _ in range(iters):
        start_event.record()
        fn()
        end_event.record()
        torch.xpu.synchronize()
        total_time_ms += start_event.elapsed_time(end_event)
    return total_time_ms / iters * 1e3


def make_benchmark_runner(
    *,
    op_name: str,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cache: torch.Tensor,
    is_neox: bool,
    benchmark_mode: str,
    kernel_buffer_count: int,
):
    if benchmark_mode == "end_to_end":

        def run() -> None:
            q = query.clone()
            k = key.clone() if key is not None else None
            getattr(torch.ops._C, op_name)(
                positions, q, k, head_size, cache, is_neox)

        return run

    q_buffers = [query.clone() for _ in range(kernel_buffer_count)]
    k_buffers = ([key.clone() for _ in range(kernel_buffer_count)]
                 if key is not None else None)
    state = {"index": 0}

    def run() -> None:
        index = state["index"]
        q = q_buffers[index]
        k = k_buffers[index] if k_buffers is not None else None
        getattr(torch.ops._C, op_name)(positions, q, k, head_size, cache, is_neox)
        state["index"] = (index + 1) % kernel_buffer_count

    return run


def make_inputs(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    rotary_dim: int,
    max_position: int,
    dtype: torch.dtype,
    is_neox: bool,
    use_key: bool,
    head_stride_is_contiguous: bool,
    device: str,
):
    rot = RotaryEmbedding(
        head_size, rotary_dim, max_position, 10000, is_neox, dtype)
    cos_sin_cache = rot.cos_sin_cache.to(device=device, dtype=dtype)
    positions = torch.randint(0, max_position, (batch_size, seq_len), device=device)

    q_shape = (batch_size, seq_len, num_heads, head_size)
    k_shape = (batch_size, seq_len, num_kv_heads, head_size)

    if head_stride_is_contiguous:
        query = torch.randn(q_shape, dtype=dtype, device=device)
        key = torch.randn(k_shape, dtype=dtype, device=device) if use_key else None
    else:
        query = torch.randn(
            (batch_size, seq_len, num_heads * 2, head_size),
            dtype=dtype,
            device=device)[:, :, ::2, :]
        key = (
            torch.randn(
                (batch_size, seq_len, num_kv_heads * 2, head_size),
                dtype=dtype,
                device=device)[:, :, ::2, :]
            if use_key else None)

    return positions, query, key, cos_sin_cache


def maybe_check_correctness(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    dtype: torch.dtype,
) -> None:
    query_ref = query.clone()
    key_ref = key.clone() if key is not None else None
    query_exp = query.clone()
    key_exp = key.clone() if key is not None else None

    torch.ops._C.rotary_embedding(
        positions, query_ref, key_ref, head_size, cos_sin_cache, is_neox)
    torch.ops._C.rotary_embedding_experimental(
        positions, query_exp, key_exp, head_size, cos_sin_cache, is_neox)

    atol = 2e-3 if dtype == torch.float16 else 5e-3
    rtol = atol
    torch.testing.assert_close(query_exp, query_ref, atol=atol, rtol=rtol)
    if key is not None:
        torch.testing.assert_close(key_exp, key_ref, atol=atol, rtol=rtol)


def print_header(include_mode: bool, include_offset: bool) -> None:
    mode_columns = f" {'mode':>21}" if include_mode else ""
    offset_columns = f" {'roff':>4}" if include_offset else ""
    print(
        f"{'dtype':<10}{mode_columns} {'bs':>3} {'seq':>6} {'nh':>4} {'nkv':>4} {'hs':>4}{offset_columns} {'rd':>4} "
        f"{'neox':>5} {'key':>4} {'contig':>6} {'ref_us':>12} {'exp_us':>12} {'exp/ref':>10}")
    width = 122 if include_mode and include_offset else 117 if include_mode else 105 if include_offset else 100
    print("-" * width)


def run_case(
    *,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    rotary_dim: int,
    max_position: int,
    dtype: torch.dtype,
    is_neox: bool,
    use_key: bool,
    head_stride_is_contiguous: bool,
    device: str,
    warmup: int,
    iters: int,
    timer: str,
    check: bool,
    benchmark_mode: str,
    kernel_buffer_count: int,
    attn_mode: str | None = None,
    rope_offset: int = 0,
    include_offset: bool = False,
):
    positions, query, key, cache = make_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position=max_position,
        dtype=dtype,
        is_neox=is_neox,
        use_key=use_key,
        head_stride_is_contiguous=head_stride_is_contiguous,
        device=device,
    )

    if check:
        maybe_check_correctness(
            positions, query, key, head_size, cache, is_neox, dtype)

    run_ref = make_benchmark_runner(
        op_name="rotary_embedding",
        positions=positions,
        query=query,
        key=key,
        head_size=head_size,
        cache=cache,
        is_neox=is_neox,
        benchmark_mode=benchmark_mode,
        kernel_buffer_count=kernel_buffer_count,
    )
    run_exp = make_benchmark_runner(
        op_name="rotary_embedding_experimental",
        positions=positions,
        query=query,
        key=key,
        head_size=head_size,
        cache=cache,
        is_neox=is_neox,
        benchmark_mode=benchmark_mode,
        kernel_buffer_count=kernel_buffer_count,
    )

    ref_us = benchmark(run_ref, warmup, iters, timer)
    exp_us = benchmark(run_exp, warmup, iters, timer)
    ratio = exp_us / ref_us if ref_us > 0 else float("inf")

    mode_columns = f" {attn_mode:>21}" if attn_mode is not None else ""
    offset_columns = f" {rope_offset:>4}" if include_offset else ""
    print(
        f"{dtype_name(dtype):<10}{mode_columns} {batch_size:>3} {seq_len:>6} {num_heads:>4} "
        f"{num_kv_heads:>4} {head_size:>4}{offset_columns} {rotary_dim:>4} {str(is_neox):>5} "
        f"{str(use_key):>4} {str(head_stride_is_contiguous):>6} "
        f"{ref_us:>12.3f} {exp_us:>12.3f} {ratio:>10.3f}")

    return {
        "dtype": dtype_name(dtype),
        "attn_mode": attn_mode or "",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
        "rope_offset": rope_offset,
        "rotary_dim": rotary_dim,
        "max_position": max_position,
        "is_neox": is_neox,
        "use_key": use_key,
        "head_stride_contiguous": head_stride_is_contiguous,
        "ref_us": ref_us,
        "exp_us": exp_us,
        "speedup": ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark rotary_embedding vs rotary_embedding_experimental")
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--batch-sizes", default="2")
    parser.add_argument("--seq-lens", default="64,256,1024")
    parser.add_argument("--num-heads", default="8,16,32")
    parser.add_argument("--num-kv-heads", default="")
    parser.add_argument("--head-sizes", default="64")
    parser.add_argument("--rotary-dims", default="64")
    parser.add_argument("--max-positions", default="4096")
    parser.add_argument("--dtypes", default="fp16,bf16")
    parser.add_argument("--is-neox", default="true,false")
    parser.add_argument("--use-key", default="true,false")
    parser.add_argument("--head-stride-contiguous", default="true,false")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--timer",
        choices=["host", "xpu_event"],
        default="host",
        help="Use host wall-clock timing or device event timing.")
    parser.add_argument(
        "--benchmark-mode",
        choices=["end_to_end", "kernel_only"],
        default="end_to_end",
        help="Benchmark end-to-end call cost or only the kernel execution with prebuilt input buffers.")
    parser.add_argument(
        "--kernel-buffer-count",
        type=int,
        default=1,
        help="Number of prebuilt input buffers to rotate through in kernel_only mode.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--csv", default="")
    parser.add_argument(
        "--client-is-neox",
        default="true,false",
        help="NeoX layouts to benchmark for the customer cases shown first.")
    parser.add_argument(
        "--skip-client-cases",
        action="store_true",
        help="Skip the pre_fa_ops-inspired customer cases at the top of the output.")
    parser.add_argument(
        "--skip-synthetic-sweep",
        action="store_true",
        help="Skip the synthetic parameter sweep section.")
    parser.add_argument(
        "--client-section",
        choices=["all", "pre_fa", "attn_ops"],
        default="all",
        help="Which customer case group to run.")
    parser.add_argument(
        "--client-case-index",
        default="all",
        help="0-based case index within the selected customer group, or 'all'.")
    args = parser.parse_args()

    batch_sizes = parse_int_list(args.batch_sizes)
    seq_lens = parse_int_list(args.seq_lens)
    num_heads_list = parse_int_list(args.num_heads)
    num_kv_heads_list = (
        parse_int_list(args.num_kv_heads) if args.num_kv_heads else num_heads_list)
    head_sizes = parse_int_list(args.head_sizes)
    rotary_dims = parse_int_list(args.rotary_dims)
    max_positions = parse_int_list(args.max_positions)
    dtypes = parse_dtype_list(args.dtypes)
    is_neox_list = parse_bool_list(args.is_neox)
    client_is_neox_list = parse_bool_list(args.client_is_neox)
    client_case_index = parse_optional_int(args.client_case_index)
    use_key_list = parse_bool_list(args.use_key)
    head_stride_list = parse_bool_list(args.head_stride_contiguous)

    if args.kernel_buffer_count < 1:
        raise ValueError("--kernel-buffer-count must be >= 1")

    csv_writer = None
    csv_file = None
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "section",
            "dtype",
            "attn_mode",
            "batch_size",
            "seq_len",
            "num_heads",
            "num_kv_heads",
            "head_size",
            "rope_offset",
            "rotary_dim",
            "max_position",
            "is_neox",
            "use_key",
            "head_stride_contiguous",
            "ref_us",
            "exp_us",
            "speedup",
        ])

    try:
        if not args.skip_client_cases:
            selected_pre_fa_cases = CLIENT_PRIORITY_CASES
            selected_attn_ops_cases = ATTN_OPS_PRIORITY_CASES
            if client_case_index is not None:
                if args.client_section == "pre_fa":
                    selected_pre_fa_cases = [CLIENT_PRIORITY_CASES[client_case_index]]
                    selected_attn_ops_cases = []
                elif args.client_section == "attn_ops":
                    selected_pre_fa_cases = []
                    selected_attn_ops_cases = [ATTN_OPS_PRIORITY_CASES[client_case_index]]
                else:
                    raise ValueError(
                        "--client-case-index requires --client-section pre_fa or attn_ops")

            if args.client_section in ("all", "pre_fa") and selected_pre_fa_cases:
                print("Client cases from pre_fa_ops.json")
                print_header(include_mode=True, include_offset=False)
                for base_case in selected_pre_fa_cases:
                    for is_neox in client_is_neox_list:
                        result = run_case(
                            batch_size=base_case["batch_size"],
                            seq_len=base_case["seq_len"],
                            num_heads=base_case["num_heads"],
                            num_kv_heads=base_case["num_kv_heads"],
                            head_size=base_case["head_size"],
                            rotary_dim=base_case["rotary_dim"],
                            max_position=max(max_positions + [base_case["seq_len"] + 1]),
                            dtype=base_case["dtype"],
                            is_neox=is_neox,
                            use_key=base_case["use_key"],
                            head_stride_is_contiguous=base_case[
                                "head_stride_is_contiguous"],
                            device=args.device,
                            warmup=args.warmup,
                            iters=args.iters,
                            timer=args.timer,
                            check=args.check,
                            benchmark_mode=args.benchmark_mode,
                            kernel_buffer_count=args.kernel_buffer_count,
                            attn_mode=base_case["attn_mode"],
                            rope_offset=0,
                            include_offset=False,
                        )
                        if csv_writer is not None:
                            csv_writer.writerow([
                                "client",
                                result["dtype"],
                                result["attn_mode"],
                                result["batch_size"],
                                result["seq_len"],
                                result["num_heads"],
                                result["num_kv_heads"],
                                result["head_size"],
                                result["rope_offset"],
                                result["rotary_dim"],
                                result["max_position"],
                                result["is_neox"],
                                result["use_key"],
                                result["head_stride_contiguous"],
                                result["ref_us"],
                                result["exp_us"],
                                result["speedup"],
                            ])
                print()

            if args.client_section in ("all", "attn_ops") and selected_attn_ops_cases:
                print("Customer cases from attn_ops.json")
                print_header(include_mode=True, include_offset=True)
                for base_case in selected_attn_ops_cases:
                    for is_neox in client_is_neox_list:
                        result = run_case(
                            batch_size=base_case["batch_size"],
                            seq_len=base_case["seq_len"],
                            num_heads=base_case["num_heads"],
                            num_kv_heads=base_case["num_kv_heads"],
                            head_size=base_case["head_size"],
                            rotary_dim=base_case["rotary_dim"],
                            max_position=max(
                                max_positions +
                                [base_case["cache_len"] + base_case["seq_len"] + 1]),
                            dtype=base_case["dtype"],
                            is_neox=is_neox,
                            use_key=base_case["use_key"],
                            head_stride_is_contiguous=base_case[
                                "head_stride_is_contiguous"],
                            device=args.device,
                            warmup=args.warmup,
                            iters=args.iters,
                            timer=args.timer,
                            check=args.check,
                            benchmark_mode=args.benchmark_mode,
                            kernel_buffer_count=args.kernel_buffer_count,
                            attn_mode=base_case["attn_mode"],
                            rope_offset=base_case["rope_offset"],
                            include_offset=True,
                        )
                        if csv_writer is not None:
                            csv_writer.writerow([
                                "customer_attn_ops",
                                result["dtype"],
                                result["attn_mode"],
                                result["batch_size"],
                                result["seq_len"],
                                result["num_heads"],
                                result["num_kv_heads"],
                                result["head_size"],
                                result["rope_offset"],
                                result["rotary_dim"],
                                result["max_position"],
                                result["is_neox"],
                                result["use_key"],
                                result["head_stride_contiguous"],
                                result["ref_us"],
                                result["exp_us"],
                                result["speedup"],
                            ])
                print()

        if not args.skip_synthetic_sweep:
            print("Synthetic sweep")
            print_header(include_mode=False, include_offset=False)
            for case in itertools.product(
                dtypes,
                batch_sizes,
                seq_lens,
                num_heads_list,
                head_sizes,
                rotary_dims,
                max_positions,
                is_neox_list,
                use_key_list,
                head_stride_list):
                (dtype, batch_size, seq_len, num_heads, head_size, rotary_dim,
                 max_position, is_neox, use_key, head_stride_is_contiguous) = case

                if rotary_dim > head_size:
                    continue

                for num_kv_heads in num_kv_heads_list:
                    if num_heads % num_kv_heads != 0:
                        continue
                    if not use_key and num_kv_heads != num_heads:
                        continue

                    result = run_case(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        head_size=head_size,
                        rotary_dim=rotary_dim,
                        max_position=max_position,
                        dtype=dtype,
                        is_neox=is_neox,
                        use_key=use_key,
                        head_stride_is_contiguous=head_stride_is_contiguous,
                        device=args.device,
                        warmup=args.warmup,
                        iters=args.iters,
                        timer=args.timer,
                        check=args.check,
                        benchmark_mode=args.benchmark_mode,
                        kernel_buffer_count=args.kernel_buffer_count,
                        rope_offset=0,
                        include_offset=False,
                    )

                    if csv_writer is not None:
                        csv_writer.writerow([
                            "grid",
                            result["dtype"],
                            result["attn_mode"],
                            result["batch_size"],
                            result["seq_len"],
                            result["num_heads"],
                            result["num_kv_heads"],
                            result["head_size"],
                            result["rope_offset"],
                            result["rotary_dim"],
                            result["max_position"],
                            result["is_neox"],
                            result["use_key"],
                            result["head_stride_contiguous"],
                            result["ref_us"],
                            result["exp_us"],
                            result["speedup"],
                        ])
    finally:
        if csv_file is not None:
            csv_file.close()


if __name__ == "__main__":
    main()