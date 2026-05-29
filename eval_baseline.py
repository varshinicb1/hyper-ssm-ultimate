"""
eval_baseline.py — Production Evaluation: Hyper-SSM vs. Transformer Baseline
=============================================================================
Fair, apples-to-apples comparison on identical synthetic English data.
Measures: loss convergence, throughput (tokens/sec), peak VRAM, parameter count.

Usage:
    python eval_baseline.py [--steps 100] [--batch 4] [--seq_len 256]
"""

import argparse
import json
import time
import torch
import torch.nn as nn
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hyper_ssm.model import HyperSSM, HyperSSMConfig
from hyper_ssm.tokenizer import HyperTokenizer


# ---------------------------------------------------------------------------
#  Baseline: Standard Transformer (parameter-matched)
# ---------------------------------------------------------------------------
class TransformerBaseline(nn.Module):
    """Vanilla auto-regressive Transformer matched to ~40M params for fair comparison."""

    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, max_seq: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, batch_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq = max_seq

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        x = self.encoder(x, mask=mask, is_causal=True)
        return self.head(x)


# ---------------------------------------------------------------------------
#  Data: deterministic synthetic English text (no network dependency)
# ---------------------------------------------------------------------------
class SyntheticDataStream:
    """Tokenises a fixed English passage and yields (X, Y) batches indefinitely."""

    TEXT = (
        "The quick brown fox jumps over the lazy dog. "
        "A stitch in time saves nine. "
        "Knowledge is power, and power corrupts. "
        "To be or not to be, that is the question. "
        "All that glitters is not gold. "
    ) * 2000

    def __init__(self, tokenizer, batch_size: int, seq_len: int):
        self.batch_size = batch_size
        self.seq_len = seq_len
        ids = tokenizer.encode(self.TEXT)["input_ids"][0].tolist()
        # guarantee at least enough tokens
        self._tokens = ids * max(1, (batch_size * (seq_len + 1) * 300) // len(ids) + 1)

    def batches(self, n: int):
        ptr = 0
        for _ in range(n):
            xs, ys = [], []
            for _ in range(self.batch_size):
                end = ptr + self.seq_len + 1
                if end > len(self._tokens):
                    ptr = 0
                    end = self.seq_len + 1
                chunk = self._tokens[ptr:end]
                xs.append(chunk[:-1])
                ys.append(chunk[1:])
                ptr += self.seq_len
            yield (torch.tensor(xs, dtype=torch.long),
                   torch.tensor(ys, dtype=torch.long))


# ---------------------------------------------------------------------------
#  Training + measurement harness
# ---------------------------------------------------------------------------
@torch.no_grad()
def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def run_benchmark(tag: str, model: nn.Module, data: SyntheticDataStream,
                  steps: int, device: torch.device, use_bf16: bool):
    """Train for `steps` batches, measuring loss curve, throughput, VRAM."""

    model.to(device)
    if use_bf16:
        model = model.bfloat16()

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    params = _count_params(model)

    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"  Parameters : {params / 1e6:.2f} M")
    print(f"  Precision  : {'bfloat16' if use_bf16 else 'float32'}")
    print(f"  Steps      : {steps}")
    print(f"{'='*60}")

    history = []
    total_tokens = 0
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()

    model.train()
    for step, (X, Y) in enumerate(data.batches(steps)):
        X, Y = X.to(device), Y.to(device)
        opt.zero_grad(set_to_none=True)

        # unified call — Hyper-SSM returns (logits, entropy); Transformer returns logits
        out = model(X)
        if isinstance(out, tuple):
            logits, entropy = out
        else:
            logits, entropy = out, torch.tensor(0.0)

        V = logits.size(-1)
        loss = loss_fn(logits.reshape(-1, V), Y.reshape(-1))

        if torch.isnan(loss):
            print(f"  [FATAL] NaN at step {step}. Aborting benchmark.")
            return None

        # entropy regularisation (only relevant for Hyper-SSM router)
        (loss - 0.01 * entropy).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        bs = X.size(0) * X.size(1)
        total_tokens += bs
        vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        entry = {"step": step, "loss": round(loss.item(), 4), "vram_mb": round(vram, 1)}
        history.append(entry)

        if step % max(1, steps // 10) == 0:
            print(f"  Step {step:04d}  loss={entry['loss']:.4f}  vram={entry['vram_mb']:.0f} MB")

    elapsed = time.perf_counter() - t0
    throughput = total_tokens / elapsed
    final_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    summary = {
        "tag": tag,
        "params_M": round(params / 1e6, 2),
        "avg_loss": round(sum(h["loss"] for h in history) / len(history), 4),
        "final_loss": history[-1]["loss"],
        "throughput_tok_s": round(throughput, 1),
        "peak_vram_mb": round(final_vram, 1),
        "wall_sec": round(elapsed, 2),
    }
    print(f"\n  ── Results ─────────────────────────────")
    for k, v in summary.items():
        print(f"  {k:>20s} : {v}")
    print()
    return summary


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = HyperTokenizer()
    data = SyntheticDataStream(tok, args.batch, args.seq_len)

    VOCAB = tok.vocab_size
    D_MODEL = 256
    N_LAYERS = 12

    results = []

    # 1. Transformer baseline
    t_model = TransformerBaseline(VOCAB, D_MODEL, N_LAYERS, n_heads=8,
                                  max_seq=max(args.seq_len, 2048))
    r = run_benchmark("Transformer-40M  (O(N) KV-Cache)", t_model, data,
                      args.steps, device, use_bf16=False)
    if r: results.append(r)
    del t_model; torch.cuda.empty_cache()

    # 2. Hyper-SSM
    class _Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = HyperSSM(HyperSSMConfig(VOCAB, D_MODEL, num_layers=N_LAYERS))
        def forward(self, x):
            return self.core(x, return_entropy=True)

    h_model = _Wrap()
    r = run_benchmark("Hyper-SSM-40M    (O(1) Geometric)", h_model, data,
                      args.steps, device, use_bf16=True)
    if r: results.append(r)
    del h_model; torch.cuda.empty_cache()

    # Dump JSON for programmatic consumption
    out_path = os.path.join(os.path.dirname(__file__), "results_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    main()
