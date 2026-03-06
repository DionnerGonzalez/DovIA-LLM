"""
DovIA v2 - Entrenamiento completo.
Uso: python scripts/train.py
"""

import os, sys, math, time, json, argparse
import torch, torch.nn as nn
from torch.optim import AdamW
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import DovIA, DovIAConfig
from src.tokenizer import BPETokenizer
from src.dataset import TextDataset, get_dataloader
from data.corpus import get_corpus, get_corpus_stats


def get_device():
    if torch.cuda.is_available():
        print(f"[DovIA] GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("[DovIA] Apple MPS")
        return torch.device("mps")
    print("[DovIA] CPU")
    return torch.device("cpu")


def cosine_lr(step, warmup, total, lr_min, lr_max):
    if step < warmup:
        return lr_max * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def train(cfg: dict):
    device = get_device()
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    # Stats del corpus
    stats = get_corpus_stats()
    print(f"\n[DovIA] Corpus cargado:")
    for k, v in stats["sections"].items():
        print(f"  {k}: {v} textos")
    print(f"  TOTAL: {stats['total_texts']} textos | {stats['total_words']:,} palabras\n")

    corpus = get_corpus()

    # Tokenizador
    tok_dir = cfg["tokenizer_dir"]
    vocab_json = os.path.join(tok_dir, "vocab.json")
    if os.path.exists(vocab_json):
        print("[DovIA] Cargando tokenizador existente...")
        tokenizer = BPETokenizer.load(tok_dir)
    else:
        tokenizer = BPETokenizer(vocab_size=cfg["vocab_size"])
        tokenizer.train(corpus, verbose=True)
        tokenizer.save(tok_dir)

    print(f"[DovIA] Vocab: {len(tokenizer)} tokens")

    # Modelo
    model_cfg = DovIAConfig(
        vocab_size=len(tokenizer),
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_kv_heads=cfg["n_kv_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
    )
    model = DovIA(model_cfg).to(device)
    print(f"[DovIA] Parámetros: {model.count_parameters():,}")

    # Dataset
    dataset = TextDataset(corpus, tokenizer,
                          context_length=cfg["context_length"],
                          stride=cfg["context_length"] // 2)
    loader = get_dataloader(dataset, batch_size=cfg["batch_size"])

    # Optimizador
    optimizer = AdamW(model.parameters(), lr=cfg["lr"],
                      weight_decay=cfg["weight_decay"], betas=(0.9, 0.95))
    total_steps = len(loader) * cfg["epochs"]
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    print(f"\n[DovIA] Iniciando entrenamiento: {cfg['epochs']} épocas | {total_steps} steps\n")

    global_step = 0
    best_loss = float("inf")

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            lr = cosine_lr(global_step, cfg["warmup_steps"], total_steps,
                           cfg["lr"] * 0.1, cfg["lr"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    _, loss = model(x, labels=y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                _, loss = model(x, labels=y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if (step + 1) % cfg["log_every"] == 0:
                avg = epoch_loss / (step + 1)
                ppl = math.exp(min(avg, 20))
                elapsed = time.time() - t0
                print(f"  E{epoch+1} | S{step+1}/{len(loader)} | "
                      f"loss={avg:.4f} ppl={ppl:.1f} lr={lr:.2e} t={elapsed:.0f}s")

        avg_loss = epoch_loss / len(loader)
        ppl = math.exp(min(avg_loss, 20))
        print(f"\n[Epoch {epoch+1}/{cfg['epochs']}] loss={avg_loss:.4f} ppl={ppl:.2f}\n")

        if avg_loss < best_loss or (epoch + 1) % cfg["save_every"] == 0:
            best_loss = min(best_loss, avg_loss)
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "model_config": vars(model_cfg),
            }
            torch.save(ckpt, os.path.join(cfg["checkpoint_dir"], "dovia_best.pt"))
            print(f"  ✅ Checkpoint guardado (loss={avg_loss:.4f})")

    print("\n[DovIA] ¡Entrenamiento completado!")
    return best_loss


DEFAULT_CONFIG = {
    "vocab_size": 6000,
    "context_length": 256,
    "d_model": 256,
    "n_heads": 8,
    "n_kv_heads": 4,
    "n_layers": 6,
    "d_ff": 1024,
    "dropout": 0.1,
    "batch_size": 8,
    "epochs": 15,
    "lr": 3e-4,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "warmup_steps": 100,
    "save_every": 5,
    "log_every": 20,
    "checkpoint_dir": "checkpoints",
    "tokenizer_dir": "checkpoints/tokenizer",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = dict(DEFAULT_CONFIG)
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg.update(json.load(f))
    train(cfg)
