"""
Benchmark: Triton Fused Attention vs PyTorch Native Attention

Compares:
- Triton fused attention (from the fused_attention tutorial)
- PyTorch scaled_dot_product_attention (SDPA) - uses best available backend
- PyTorch manual attention (matmul + softmax + matmul)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fused_attention'))

import torch
import triton

from fused_att import attention as triton_attention, is_hip, is_cuda, is_blackwell, is_hopper


DEVICE = triton.runtime.driver.active.get_active_torch_device()


def pytorch_sdpa(q, k, v, causal, sm_scale):
    """PyTorch scaled_dot_product_attention (picks best backend automatically)."""
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal, scale=sm_scale,
    )


def pytorch_manual(q, k, v, causal, sm_scale):
    """Manual PyTorch attention: QK^T * scale -> softmax -> * V."""
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        N_CTX = q.shape[2]
        M = torch.tril(torch.ones((N_CTX, N_CTX), device=q.device, dtype=torch.bool))
        p = p.masked_fill(~M, float("-inf"))
    p = torch.softmax(p.float(), dim=-1).to(q.dtype)
    return torch.matmul(p, v)


BATCH, N_HEADS = 4, 32

configs = []
for HEAD_DIM in [64, 128]:
    for mode in ["fwd", "bwd"]:
        for causal in [True, False]:
            configs.append(
                triton.testing.Benchmark(
                    x_names=["N_CTX"],
                    x_vals=[2**i for i in range(10, 15)],
                    line_arg="provider",
                    line_vals=["triton", "pytorch-sdpa", "pytorch-manual"],
                    line_names=["Triton Fused", "PyTorch SDPA", "PyTorch Manual"],
                    styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                    ylabel="TFLOPS",
                    plot_name=f"attention-bench-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}",
                    args={
                        "H": N_HEADS,
                        "BATCH": BATCH,
                        "HEAD_DIM": HEAD_DIM,
                        "mode": mode,
                        "causal": causal,
                    },
                ))


@triton.testing.perf_report(configs)
def bench_attention(BATCH, H, N_CTX, HEAD_DIM, causal, mode, provider, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    sm_scale = 0.5

    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)

    if provider == "triton":
        fn = lambda: triton_attention(q, k, v, causal, sm_scale, False)
    elif provider == "pytorch-sdpa":
        fn = lambda: pytorch_sdpa(q, k, v, causal, sm_scale)
    elif provider == "pytorch-manual":
        # Skip manual attention for large sequences to avoid OOM
        if N_CTX > 8192:
            return float("nan")
        fn = lambda: pytorch_manual(q, k, v, causal, sm_scale)

    if mode == "bwd":
        o = fn()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)

    ms = triton.testing.do_bench(fn)

    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    bench_attention.run(save_path=".", print_data=True)
