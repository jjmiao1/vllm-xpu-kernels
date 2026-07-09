# SPDX-License-Identifier: Apache-2.0
"""Correctness test for the BF16xFP32 emulated-FP32 GEMM.

The kernel emulates an FP32-precision GEMM (BF16 activation x FP32 weight)
by decomposing the FP32 weight into two BF16 matrices and running a DualGemm:

    out = A @ B_high.T + (A @ B_low.T) * scale

This mirrors the way the Hunyuan-V3 MoE router (GateLinear) is computed on
XPU today.  The result is compared against the reference ``F.linear`` path
that vLLM currently uses on XPU:

        x_fp32 = x_bf16.float()
        out    = F.linear(x_fp32, W_fp32)      # FP32 x FP32 GEMM

Run:
    pytest -q tests/test_gemm_bf16xfp32.py
"""

import pytest
import torch

from tests.register_ops import gemm_bf16xfp32

DEVICE = "xpu"
SCALE = 256.0

# (M, N, K)
#   M = num_tokens (decode: small, prefill: large)
#   N = num_experts (Hunyuan-V3 router = 192)
#   K = hidden_size (Hunyuan-V3 = 4096)
MNK_FACTORS = [
    # decode-ish (small M) -> policy_small_m / policy_medium_m
    (1, 192, 4096),
    (8, 192, 4096),
    (16, 192, 4096),
    (32, 192, 4096),
    (64, 192, 4096),
    # prefill-ish (large M) -> policy_default
    (256, 192, 4096),
    (1024, 192, 4096),
    (4096, 192, 4096),
    # a few square-ish shapes to exercise other tiles
    (256, 256, 2048),
    (512, 512, 4096),
]


def _split_fp32_weight(weight: torch.Tensor, scale: float = SCALE):
    """FP32 weight [N, K] -> (W_high, W_low) BF16 decomposition.

    The kernel computes ``Y[m,n] = sum_k A[m,k] * B[k,n]`` (i.e. F.linear),
    so the components are returned transposed to ``[K, N]`` (N-contiguous),
    matching the layout the kernel expects.

    Reconstruction: weight ~= fp32(W_high.T) + fp32(W_low.T) * scale
    """
    assert weight.dtype == torch.float32
    W_high = weight.to(torch.bfloat16)
    residual = weight - W_high.float()
    W_low = (residual / scale).to(torch.bfloat16)
    return W_high.t().contiguous(), W_low.t().contiguous()


def _make_inputs(m, n, k, seed=1234):
    torch.manual_seed(seed)
    # activation is genuinely BF16 (as produced by the previous layer)
    x_bf16 = (torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE) / 8.0)
    # weight is a genuine FP32 tensor (router gate.weight)
    w_fp32 = (torch.randn(n, k, dtype=torch.float32, device=DEVICE) / 8.0)
    return x_bf16, w_fp32


def _ref_flinear(x_bf16, w_fp32):
    """vLLM XPU Tier-4 fallback: cast input to FP32, FP32xFP32 GEMM."""
    return torch.nn.functional.linear(x_bf16.float(), w_fp32)


def _ref_bf16_naive(x_bf16, w_fp32):
    """Single BF16 GEMM (weight truncated to BF16), FP32 accumulate.

    Used only to show that the DualGemm result is *closer* to full FP32
    than a plain BF16 truncation would be.
    """
    w_high = w_fp32.to(torch.bfloat16)
    return torch.nn.functional.linear(x_bf16, w_high).float()


@pytest.mark.parametrize("mnk", MNK_FACTORS)
def test_gemm_bf16xfp32_correctness(mnk):
    m, n, k = mnk
    x_bf16, w_fp32 = _make_inputs(m, n, k)

    # ground truth: full FP32 compute
    ref = _ref_flinear(x_bf16, w_fp32)

    # kernel under test
    w_high, w_low = _split_fp32_weight(w_fp32, SCALE)
    out = gemm_bf16xfp32(x_bf16, w_high, w_low, SCALE)
    torch.xpu.synchronize()

    assert out.dtype == torch.float32
    assert out.shape == ref.shape

    # The DualGemm emulates FP32 using two BF16 DPAS ops. Because the
    # *activation* is BF16, the achievable precision is ~ BF16-input x
    # FP32-weight. We compare against the FP32 reference with a tolerance
    # dominated by the BF16 activation rounding (~2^-8 relative).
    abs_err = (out - ref).abs()
    rel_err = abs_err / (ref.abs() + 1e-6)

    # dual-bf16 should be dramatically better than naive single-bf16 weight
    naive = _ref_bf16_naive(x_bf16, w_fp32)
    naive_max_rel = ((naive - ref).abs() / (ref.abs() + 1e-6)).max().item()
    dual_max_rel = rel_err.max().item()

    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
    # the emulation must not be worse than the naive bf16-weight truncation
    assert dual_max_rel <= naive_max_rel + 1e-3, (
        f"dual-bf16 rel err {dual_max_rel:.4e} worse than "
        f"naive bf16 {naive_max_rel:.4e}")


if __name__ == "__main__":
    # allow running as a plain script for quick manual checks
    for mnk in MNK_FACTORS:
        test_gemm_bf16xfp32_correctness(mnk)
    print("\nAll gemm_bf16xfp32 correctness tests passed.")
