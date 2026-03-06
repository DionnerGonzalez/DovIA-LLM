"""
DovIA v2 - Tokenizador BPE avanzado con soporte completo para español.
"""

import os, re, json
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


SPECIAL_TOKENS = {
    "<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
    "<mask>": 4, "<sep>": 5, "<user>": 6, "<dovia>": 7,
}

# Patrón GPT-4 adaptado para español
PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d"""
    r"""| ?[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+"""
    r"""| ?[0-9]+"""
    r"""| ?[^\s\w]+"""
    r"""|\s+(?!\S)|\s+""",
    re.UNICODE
)


class BPETokenizer:
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: Dict[Tuple[str, str], str] = {}
        self.merge_order: List[Tuple[str, str]] = []

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    def _word_freqs(self, corpus: List[str]) -> Dict[str, int]:
        freq: Dict[str, int] = defaultdict(int)
        for text in corpus:
            for word in re.findall(PATTERN, text):
                key = " ".join(list(word.encode("utf-8").decode("utf-8"))) + " </w>"
                freq[key] += 1
        return dict(freq)

    def _get_pairs(self, wfreq: Dict[str, int]) -> Dict[Tuple[str,str], int]:
        pairs: Dict[Tuple[str,str], int] = defaultdict(int)
        for word, freq in wfreq.items():
            syms = word.split()
            for i in range(len(syms)-1):
                pairs[(syms[i], syms[i+1])] += freq
        return dict(pairs)

    def _apply_merge(self, pair: Tuple[str,str], wfreq: Dict[str,int]) -> Dict[str,int]:
        a, b = re.escape(pair[0]), re.escape(pair[1])
        pattern = re.compile(r"(?<!\S)" + a + r" " + b + r"(?!\S)")
        merged = pair[0] + pair[1]
        return {pattern.sub(merged, w): f for w, f in wfreq.items()}

    def train(self, corpus: List[str], verbose: bool = True):
        self.vocab = dict(SPECIAL_TOKENS)
        nid = len(self.vocab)
        # Bytes base
        for i in range(256):
            ch = chr(i)
            if ch not in self.vocab:
                self.vocab[ch] = nid; nid += 1
        # Caracteres especiales del español
        for ch in "áéíóúüñÁÉÍÓÚÜÑ¿¡":
            if ch not in self.vocab:
                self.vocab[ch] = nid; nid += 1
        # Marcador de fin de palabra
        if "</w>" not in self.vocab:
            self.vocab["</w>"] = nid; nid += 1

        wfreq = self._word_freqs(corpus)
        n_merges = self.vocab_size - nid
        if verbose:
            print(f"[DovIA Tokenizer] Entrenando {n_merges} merges sobre {len(corpus)} textos...")

        for i in range(n_merges):
            pairs = self._get_pairs(wfreq)
            if not pairs: break
            best = max(pairs, key=pairs.get)
            wfreq = self._apply_merge(best, wfreq)
            merged = best[0] + best[1]
            self.merges[best] = merged
            self.merge_order.append(best)
            if merged not in self.vocab:
                self.vocab[merged] = nid; nid += 1
            if verbose and (i+1) % 500 == 0:
                print(f"  [{i+1}/{n_merges}] vocab={nid}")

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        if verbose:
            print(f"[DovIA Tokenizer] Listo. Vocab final: {len(self.vocab)} tokens")

    # ── Encode / Decode ───────────────────────────────────────────────────────
    def _bpe_word(self, word: str) -> List[str]:
        syms = list(word) + ["</w>"]
        while True:
            pairs = [(syms[i], syms[i+1]) for i in range(len(syms)-1)]
            candidates = [p for p in pairs if p in self.merges]
            if not candidates: break
            best = min(candidates, key=lambda p: self.merge_order.index(p))
            merged = self.merges[best]
            new = []
            i = 0
            while i < len(syms):
                if i < len(syms)-1 and (syms[i], syms[i+1]) == best:
                    new.append(merged); i += 2
                else:
                    new.append(syms[i]); i += 1
            syms = new
        return syms

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = []
        if add_special_tokens:
            ids.append(SPECIAL_TOKENS["<bos>"])
        for word in re.findall(PATTERN, text):
            for tok in self._bpe_word(word):
                ids.append(self.vocab.get(tok, SPECIAL_TOKENS["<unk>"]))
        if add_special_tokens:
            ids.append(SPECIAL_TOKENS["<eos>"])
        return ids

    def encode_chat(self, user_msg: str) -> List[int]:
        """Formato conversacional: <user> mensaje <dovia>"""
        ids = [SPECIAL_TOKENS["<bos>"], SPECIAL_TOKENS["<user>"]]
        ids += self.encode(user_msg, add_special_tokens=False)
        ids += [SPECIAL_TOKENS["<sep>"], SPECIAL_TOKENS["<dovia>"]]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        skip_set = set(SPECIAL_TOKENS.values()) if skip_special else set()
        toks = [self.inverse_vocab.get(i, "") for i in ids if i not in skip_set]
        return "".join(toks).replace("</w>", " ").strip()

    # ── Persistencia ─────────────────────────────────────────────────────────
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        with open(os.path.join(path, "merges.json"), "w", encoding="utf-8") as f:
            json.dump([[a, b] for a, b in self.merge_order], f, ensure_ascii=False)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({"vocab_size": self.vocab_size, "special_tokens": SPECIAL_TOKENS}, f, indent=2)
        print(f"[DovIA Tokenizer] Guardado en {path}")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(os.path.join(path, "vocab.json"), encoding="utf-8") as f:
            vocab = json.load(f)
        with open(os.path.join(path, "merges.json"), encoding="utf-8") as f:
            merge_list = json.load(f)
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        tok = cls(vocab_size=cfg["vocab_size"])
        tok.vocab = {k: int(v) for k, v in vocab.items()}
        tok.inverse_vocab = {int(v): k for k, v in vocab.items()}
        tok.merge_order = [(m[0], m[1]) for m in merge_list]
        tok.merges = {(m[0], m[1]): m[0]+m[1] for m in merge_list}
        return tok

    def __len__(self):
        return len(self.vocab)
