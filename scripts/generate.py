"""
DovIA v2 - Generación de texto y modo chat.
Uso:
  python scripts/generate.py --prompt "¿Qué es Cuba?"
  python scripts/generate.py --chat
"""

import os, sys, argparse, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import DovIA, DovIAConfig
from src.tokenizer import BPETokenizer


def load(ckpt_path: str, tok_dir: str, device):
    tokenizer = BPETokenizer.load(tok_dir)
    ckpt = torch.load(ckpt_path, map_location=device)
    mc = ckpt["model_config"]
    cfg = DovIAConfig(
        vocab_size=mc.get("vocab_size", len(tokenizer)),
        context_length=mc.get("context_length", 256),
        d_model=mc.get("d_model", 256),
        n_heads=mc.get("n_heads", 8),
        n_kv_heads=mc.get("n_kv_heads", 4),
        n_layers=mc.get("n_layers", 6),
        d_ff=mc.get("d_ff", 1024),
        dropout=0.0,
    )
    model = DovIA(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[DovIA] Modelo listo | Epoch {ckpt.get('epoch','?')} | Loss {ckpt.get('loss',0):.4f}")
    return model, tokenizer


def gen(model, tokenizer, prompt, device, max_tokens=180,
        temperature=0.82, top_k=50, top_p=0.92, rep_penalty=1.15):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    inp = torch.tensor([ids], dtype=torch.long).to(device)
    out = model.generate(inp, max_new_tokens=max_tokens, temperature=temperature,
                         top_k=top_k, top_p=top_p, repetition_penalty=rep_penalty,
                         eos_token_id=2)
    new_ids = out[0, len(ids):].tolist()
    return tokenizer.decode(new_ids, skip_special=True)


def chat_loop(model, tokenizer, device, **kw):
    print("\n" + "="*58)
    print("  🤖 DovIA v2 — Chat")
    print("  Escribe tu pregunta. 'salir' para terminar.")
    print("="*58 + "\n")
    history = ""
    while True:
        try:
            user = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[DovIA] ¡Hasta luego!"); break
        if not user: continue
        if user.lower() in ("salir","exit","quit"):
            print("[DovIA] ¡Hasta luego!"); break

        # Contexto acumulado
        prompt = history + f"Humano: {user}\nDovIA:"
        response = gen(model, tokenizer, prompt, device, **kw)
        # Limpiar respuesta hasta el siguiente marcador
        for stop in ["Humano:", "\n\n"]:
            if stop in response:
                response = response[:response.index(stop)]
        response = response.strip()
        print(f"\nDovIA: {response}\n")
        # Acumular contexto (últimas 2 rondas)
        history += f"Humano: {user}\nDovIA: {response}\n"
        lines = history.split("\n")
        if len(lines) > 12:
            history = "\n".join(lines[-12:]) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="¿Qué es Cuba?")
    parser.add_argument("--checkpoint", default="checkpoints/dovia_best.pt")
    parser.add_argument("--tokenizer_dir", default="checkpoints/tokenizer")
    parser.add_argument("--max_tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.82)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.92)
    parser.add_argument("--rep_penalty", type=float, default=1.15)
    parser.add_argument("--chat", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load(args.checkpoint, args.tokenizer_dir, device)

    kw = dict(max_tokens=args.max_tokens, temperature=args.temperature,
              top_k=args.top_k, top_p=args.top_p, rep_penalty=args.rep_penalty)

    if args.chat:
        chat_loop(model, tokenizer, device, **kw)
    else:
        response = gen(model, tokenizer, args.prompt, device, **kw)
        print(f"\nDovIA: {response}")


if __name__ == "__main__":
    main()
