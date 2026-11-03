import timeit, math
import argparse
import statistics
from contextlib import nullcontext

import torch
import torch.cuda.nvtx as nvtx

import cs336_basics.model as basics_model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy, softmax


MODEL_SIZES = {
    "small": dict(d_model = 768, d_ff = 3072, num_layers = 12, num_heads = 12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}

def nvtx_range(name: str, device: str):
    # NVTX only works with CUDA; no-op elsewhere (e.g. mps / cpu on mac)
    return torch.cuda.nvtx.range(name) if device == "cuda" else nullcontext()

def amp_ctx(device: str, is_mixed_precision: bool):
    if not is_mixed_precision:
        return nullcontext()
    return torch.autocast(device_type=device, dtype=torch.bfloat16)

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
    with nvtx.range("computing softmax"):
        weights = softmax(scores, dim=-1)
    with nvtx.range("final matmul"):
        out = torch.matmul(weights, V)
    return out

# basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

def step(model, optimizer, x, y, args):
    device = x.device.type
    t_fwd, t_bwd, t_opt = 0, 0, 0
    optimizer.zero_grad(set_to_none = True)
    sync(device)
    t0 = timeit.default_timer()
    if args.forward_only:
        with nvtx_range("forward", device), torch.no_grad(), amp_ctx(device, args.mixed_precision):
            model(x)
            sync(device)
        t_fwd = timeit.default_timer() - t0
        return t_fwd, t_bwd, t_opt

    with nvtx_range("forward", device), amp_ctx(device, args.mixed_precision):
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
        sync(device)
    t1 = timeit.default_timer()

    with nvtx_range("backward", device):
        loss.backward()
        sync(device)
    t2 = timeit.default_timer()

    with nvtx_range("optimizer", device):
        optimizer.step()
        sync(device)
    t3 = timeit.default_timer()

    t_fwd, t_bwd, t_opt = t1 - t0, t2 - t1, t3 - t2
    return t_fwd, t_bwd, t_opt

def report(records, args):
    def stats(vals):
        ms = [v * 1000 for v in vals]
        return statistics.mean(ms), (statistics.stdev(ms) if len(ms) > 1 else 0.0)
 
    mode = "forward-only (no_grad)" if args.forward_only else "train step"
    prec = "bf16-autocast" if args.mixed_precision else "fp32"
    print(f"[{args.model_size} | ctx={args.context_length} | {mode} | {prec} | "
          f"warmup={args.warmup_steps} steps={args.bench_steps}]")
 
    if args.forward_only:
        parts = {"forward": stats([r[0] for r in records])}
    else:
        parts = {
            "forward":          stats([r[0] for r in records]),
            "backward":         stats([r[1] for r in records]),
            "optimizer":        stats([r[2] for r in records]),
            "fwd+bwd":          stats([r[0] + r[1] for r in records]),
            "full train step":  stats([sum(r) for r in records]),
        }
    for name, (mean, std) in parts.items():
        print(f"  {name:<16} {mean:9.2f} ± {std:7.2f} ms")


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    return

def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--forward_only", action = "store_true", help = "forward only: pure inference under torch.no_grad();")
    p.add_argument("--mixed_precision", action="store_true",
        help="run forward (and loss) under torch.autocast with bf16")
    p.add_argument("--model_size", type = str, choices=MODEL_SIZES.keys(), default = "small")
    p.add_argument("--vocab_size", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--rope_theta", type=float, default=10_000.0)
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--bench_steps", type=int, default=10)
    args = p.parse_args()

    device = default_device()
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), dtype = torch.long, device = device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), dtype = torch.long, device = device)
    model_cfg = MODEL_SIZES[args.model_size]
    model = BasicsTransformerLM(
        vocab_size = args.vocab_size, 
        context_length = args.context_length,
        rope_theta = args.rope_theta,
        **model_cfg,
    ).to(device)
    optimizer = AdamW(
        params = model.parameters(),
        lr = 1e-3,
        betas = (0.9, 0.999),
        weight_decay = 0.01,
    )

    for _ in range(args.warmup_steps):
        step(model, optimizer, x, y, args)

    records = []
    with nvtx_range("measure", device):
        for i in range(args.bench_steps):
            with nvtx_range(f"step {i}", device):
                records.append(step(model, optimizer, x, y, args))

    report(records, args)

if __name__ == "__main__":
    main()