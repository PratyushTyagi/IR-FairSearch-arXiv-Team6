"""
encoders.py
Two sentence encoders behind one interface.

  - MiniLM   : all-MiniLM-L6-v2 via sentence-transformers (384-d, fast).
  - SPECTER2 : allenai/specter2_base + proximity adapter (768-d, scientific).
               Follows AllenAI's documented usage: encode "title[SEP]abstract"
               and take the [CLS] token of the last hidden state.

All outputs are L2-normalized so cosine similarity == dot product, which
matches the cosine space we configure in ChromaDB.
"""
import numpy as np
import config as C


def _normalize(x):
    x = np.asarray(x, dtype="float32")
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


class MiniLMEncoder:
    name = "minilm"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(C.MINILM_MODEL)

    def encode(self, titles, abstracts):
        texts = [f"{t}. {a}" for t, a in zip(titles, abstracts)]
        emb = self.model.encode(
            texts, batch_size=C.EMBED_BATCH, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return emb.astype("float32")


class Specter2Encoder:
    name = "specter2"

    def __init__(self):
        import torch
        from transformers import AutoTokenizer
        from adapters import AutoAdapterModel   # pip install adapters
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(C.SPECTER2_BASE)
        self.model = AutoAdapterModel.from_pretrained(C.SPECTER2_BASE)
        self.model.load_adapter(C.SPECTER2_ADAPTER, source="hf",
                                load_as="proximity", set_active=True)
        self.model.to(self.device).eval()

    def encode(self, titles, abstracts):
        sep = self.tok.sep_token
        texts = [f"{t}{sep}{a}" for t, a in zip(titles, abstracts)]
        out = []
        for i in range(0, len(texts), C.EMBED_BATCH):
            batch = texts[i:i + C.EMBED_BATCH]
            inp = self.tok(batch, padding=True, truncation=True,
                           max_length=512, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                hidden = self.model(**inp).last_hidden_state
            out.append(hidden[:, 0, :].cpu().numpy())   # CLS token
        return _normalize(np.vstack(out))


def get_encoder(name):
    return {"minilm": MiniLMEncoder, "specter2": Specter2Encoder}[name]()
