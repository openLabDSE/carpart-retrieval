"""Part-type identifier for the prototype (thesis model D, with the A6 "I don't know" rule).

Embeds a photo with the blended CLIP (indexes/typeproto/model_blend.pt), finds the most similar reference photo of
every part type, and answers only when the best similarity is above the threshold of the chosen strictness mode.
Built by scripts/proto/build_type_prototype.py. The model is loaded on first use.
"""
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
MODES = ("careful", "balanced", "guess")  # "guess" never refuses: it shows the best match, flagged as unconfirmed


def rank_types(vec: np.ndarray, emb: np.ndarray, types: np.ndarray) -> list:
    """(type, best similarity) for every type, highest first. Vectors are L2-normalised, so the product is the cosine."""
    sims = emb @ vec
    best = {}
    for t, s in zip(types.tolist(), sims.tolist()):
        if s > best.get(t, -1.0):
            best[t] = s
    return sorted(best.items(), key=lambda kv: -kv[1])


def decide(ranked: list, threshold: float, mode: str, top_n: int = 3) -> dict:
    """Answer when the best score is above the threshold (a score equal to it is refused, as in A6)."""
    top = [{"type": t, "score": round(s, 4)} for t, s in ranked[:top_n]]
    answered = bool(ranked) and ranked[0][1] > threshold
    return {"answered": answered, "type": ranked[0][0] if answered else None,
            "score": top[0]["score"] if top else None, "threshold": threshold, "mode": mode, "top": top}


class TypeIdentifier:
    def __init__(self, emb: np.ndarray, types: np.ndarray, thresholds: dict, embed_fn=None, model_dir: Path | None = None):
        self.emb, self.types, self.thresholds = emb, types, thresholds
        self._embed_fn, self._model_dir = embed_fn, model_dir
        self._model = None

    @classmethod
    def load(cls, model_dir: Path) -> "TypeIdentifier":
        g = np.load(model_dir / "gallery.npz")
        meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        return cls(g["emb"], g["types"], {**meta["thresholds"], "guess": -1.0}, model_dir=model_dir)

    def _load_model(self):
        import torch
        from torchvision import transforms as T
        from transformers import CLIPVisionModelWithProjection

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
        state = torch.load(self._model_dir / "model_blend.pt", map_location="cpu", weights_only=True)
        with torch.no_grad():
            model.vision_model.load_state_dict({k: v.to(dev) for k, v in state["vision_model"].items()})
            model.visual_projection.load_state_dict({k: v.to(dev) for k, v in state["visual_projection"].items()})
        prep = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.PILToTensor()])
        self._model = (model, state["proj"]["weight"].to(dev), prep, dev, torch)
        logger.info(f"Type identifier model loaded on {dev}")

    def embed(self, image) -> np.ndarray:
        if self._embed_fn is not None:
            return self._embed_fn(image)
        if self._model is None:
            self._load_model()
        model, proj, prep, dev, torch = self._model
        mean = torch.tensor(CLIP_MEAN, device=dev).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=dev).view(1, 3, 1, 1)
        x = prep(image.convert("RGB")).unsqueeze(0).to(dev).float() / 255
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            f = model.visual_projection(model.vision_model(pixel_values=(x - mean) / std).pooler_output)
            e = torch.nn.functional.normalize((f.float() @ proj.T).float(), dim=-1)
        return e[0].cpu().numpy()

    def identify(self, image, mode: str = "balanced") -> dict:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        return decide(rank_types(self.embed(image), self.emb, self.types), self.thresholds[mode], mode)
