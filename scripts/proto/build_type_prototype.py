#!/usr/bin/env python3
"""Build the assets for the part-type prototype (demo for the garage owner).

Model D of A5 (thesis): the fine-tuned model B blended 50/50 with frozen CLIP (WiSE-FT, alpha 0.5), seed 42.
Reference photos: the type_v2 gallery (every type, including the 5 that were held out of training).
Thresholds: read from the A6 evaluation of the same model (Experiments/A6 Results on type_v2), so the "I don't know"
rule is the one that was measured. Writes to indexes/typeproto/ (git-ignored):
  model_blend.pt   blended weights (vision tower, CLIP projection, 768x768 head)
  gallery.npz      L2-normalised embeddings, part types, source image paths
  meta.json        model description, thresholds, counts

Usage: python scripts/proto/build_type_prototype.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from transformers import CLIPVisionModelWithProjection

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from eval_checkpoint import CLIP_MEAN, CLIP_STD  # noqa: E402
from wise_ft import blend, frozen_state  # noqa: E402

E = Path(r"E:\car_search_data")
CKPT = E / "checkpoints/a5seeds_v2/B_s42/model.pt"
A6 = E / "results/a6/B_s42/a6_a0.50.json"
EVAL = E / "eda/type_v2/splits_type_v2_eval.json"
OUT = REPO / "indexes/typeproto"
ALPHA = 0.5


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
    frozen = frozen_state(model)
    tuned = torch.load(CKPT, map_location="cpu", weights_only=True)
    state = blend(frozen, tuned, ALPHA)
    torch.save(state, OUT / "model_blend.pt")
    with torch.no_grad():
        model.vision_model.load_state_dict({k: v.to(dev) for k, v in state["vision_model"].items()})
        model.visual_projection.load_state_dict({k: v.to(dev) for k, v in state["visual_projection"].items()})
    proj = state["proj"]["weight"].to(dev)

    gallery = json.loads(EVAL.read_text(encoding="utf-8"))["catalog"]
    prep = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.PILToTensor()])
    mean = torch.tensor(CLIP_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=dev).view(1, 3, 1, 1)
    embs = []
    for i in range(0, len(gallery), 64):
        chunk = gallery[i:i + 64]
        x = torch.stack([prep(Image.open(r["image"]).convert("RGB")) for r in chunk]).to(dev).float() / 255
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            f = model.visual_projection(model.vision_model(pixel_values=(x - mean) / std).pooler_output)
            e = torch.nn.functional.normalize((f.float() @ proj.T).float(), dim=-1).cpu().numpy()
        embs.append(e)
        print(f"embedded {min(i + 64, len(gallery))}/{len(gallery)}", flush=True)
    emb = np.concatenate(embs).astype(np.float32)
    types = [r["part_id"] for r in gallery]
    np.savez_compressed(OUT / "gallery.npz", emb=emb, types=np.array(types), paths=np.array([r["image"] for r in gallery]))

    a6 = json.loads(A6.read_text(encoding="utf-8"))
    meta = {"model": "A5 model D: fine-tuned B (seed 42) blended 50/50 with frozen CLIP ViT-L/14",
            "thresholds": {"careful": a6["calibrations"]["near"]["threshold"],
                           "balanced": a6["calibrations"]["all"]["threshold"]},
            "threshold_source": str(A6), "gallery_images": len(types), "types": sorted(set(types)),
            "note": "Thresholds were set on a gallery without the 5 held-out types; prototype only."}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"done: {len(types)} reference photos, {len(meta['types'])} types, thresholds {meta['thresholds']}")


if __name__ == "__main__":
    main()
