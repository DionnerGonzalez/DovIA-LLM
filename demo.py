"""
DovIA v2 — Demo: entrena con todo el corpus y luego abre el chat.
Uso: python demo.py
"""

import os, sys, math, time, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.model import DovIA, DovIAConfig
from src.tokenizer import BPETokenizer
from src.dataset import TextDataset, get_dataloader
from data.corpus import get_corpus, get_corpus_stats


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n{'='*58}")
    print("  🤖  DovIA v2 — Demo completa")
    print(f"{'='*58}")
    print(f"  Dispositivo: {device}")

    stats = get_corpus_stats()
    print(f"  Corpus: {stats['total_texts']} textos | {stats['total_words']:,} palabras")
    for k, v in stats["sections"].items():
        print(f"    · {k}: {v} textos")
    print()

    corpus = get_corpus()

    # ── Tokenizador ───────────────────────────────────────────────────────────
    os.makedirs("checkpoints/tokenizer", exist_ok=True)
    tok_dir = "checkpoints/tokenizer"
    if os.path.exists(os.path.join(tok_dir, "vocab.json")):
        print("[DovIA] Cargando tokenizador guardado...")
        tokenizer = BPETokenizer.load(tok_dir)
    else:
        tokenizer = BPETokenizer(vocab_size=4000)
        tokenizer.train(corpus, verbose=True)
        tokenizer.save(tok_dir)
    print(f"[DovIA] Vocab: {len(tokenizer)} tokens\n")

    # ── Modelo ────────────────────────────────────────────────────────────────
    CFG = DovIAConfig(
        vocab_size=len(tokenizer),
        context_length=192,
        d_model=256,
        n_heads=8,
        n_kv_heads=4,
        n_layers=6,
        d_ff=1024,
        dropout=0.1,
    )
    model = DovIA(CFG).to(device)
    print(f"[DovIA] Parámetros: {model.count_parameters():,}\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = TextDataset(corpus, tokenizer, context_length=192, stride=96)
    loader  = get_dataloader(dataset, batch_size=8)

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    EPOCHS = 12
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    total_steps = EPOCHS * len(loader)
    global_step = 0
    best_loss = float("inf")

    print(f"[DovIA] Entrenando {EPOCHS} épocas × {len(loader)} pasos...\n")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            # Cosine LR con warmup
            warmup = 150
            if global_step < warmup:
                lr = 3e-4 * global_step / max(1, warmup)
            else:
                progress = (global_step - warmup) / max(1, total_steps - warmup)
                lr = 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            _, loss = model(x, labels=y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            global_step += 1

        avg = epoch_loss / len(loader)
        ppl = math.exp(min(avg, 20))
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:2d}/{EPOCHS} | loss={avg:.4f} | ppl={ppl:.1f} | {elapsed:.0f}s")

        if avg < best_loss:
            best_loss = avg
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "loss": avg,
                "model_config": vars(CFG),
            }, "checkpoints/dovia_best.pt")

    print(f"\n[DovIA] ✅ Entrenamiento listo. Mejor loss: {best_loss:.4f}\n")

    # ── Chat ──────────────────────────────────────────────────────────────────
    model.eval()
    print("="*58)
    print("  💬  DovIA v2 — Chat interactivo")
    print("  Escribe cualquier pregunta. 'salir' para terminar.")
    print("="*58 + "\n")

    history = ""
    while True:
        try:
            user = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[DovIA] ¡Hasta luego!")
            break
        if not user:
            continue
        if user.lower() in ("salir", "exit", "quit"):
            print("[DovIA] ¡Hasta luego!")
            break

        prompt = history + f"Humano: {user}\nDovIA:"
        ids    = tokenizer.encode(prompt, add_special_tokens=False)
        inp    = torch.tensor([ids], dtype=torch.long).to(device)

        with torch.no_grad():
            out = model.generate(inp, max_new_tokens=180, temperature=0.82,
                                 top_k=50, top_p=0.92, repetition_penalty=1.15)
        new_ids  = out[0, len(ids):].tolist()
        response = tokenizer.decode(new_ids, skip_special=True)
        for stop in ["Humano:", "\n\n"]:
            if stop in response:
                response = response[:response.index(stop)]
        response = response.strip()
        if not response:
            response = "Interesante pregunta. Con más entrenamiento podré responderte mejor."

        print(f"\nDovIA: {response}\n")
        history += f"Humano: {user}\nDovIA: {response}\n"
        lines = history.split("\n")
        if len(lines) > 14:
            history = "\n".join(lines[-14:]) + "\n"


if __name__ == "__main__":
    run()
