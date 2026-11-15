#!/usr/bin/env python3
"""A6 abstention: can the system say "not in the catalogue" for parts it does not know?

Gallery = type_v2 gallery WITHOUT the 5 held-out types (they become unknown car parts). Known queries = the
type_v2 seen-type queries. Unknowns = data/processed/queries_unanswerable_v1.json (near = car parts, far = motorcycle).
Score of a query = its highest cosine similarity to any gallery image. A query is ANSWERED when score > threshold.

Rule (decided 2026-11-12): the threshold is the lowest one with FAR <= 5% on the VALIDATION unknowns
(FAR = share of unknown parts that still get an answer). Everything is then reported on the TEST unknowns, per type,
together with the false abstention rate (known queries refused) at that threshold. Two calibrations are reported:
  near : FAR <= 5% on validation NEAR unknowns (primary: the hard, realistic case)
  all  : FAR <= 5% on validation near + far unknowns pooled (secondary: far unknowns are easy, so this is looser)
The threshold never sees test unknowns or the known queries. Also saved: risk-coverage curve on known queries
(selective accuracy, Geifman and El-Yaniv 2017), AUROC known vs unknown, and a threshold sweep (curve.csv).

Usage:
    python scripts/eval_a6.py --checkpoint frozen --out E:/car_search_data/results/a6/A_frozen
    python scripts/eval_a6.py --checkpoint E:/car_search_data/checkpoints/a5seeds_v2/C_s42/model.pt --alphas 0.5,1 --out E:/car_search_data/results/a6/C_s42
    python scripts/eval_a6.py --checkpoint frozen --smoke --device cpu --out <tmp>    # 100-image plumbing check
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_a5 import type_metrics
from eval_checkpoint import CLIP_MEAN, CLIP_STD, bootstrap_ci
from wise_ft import blend, frozen_state

REPO = Path(__file__).resolve().parent.parent
EDA = Path(r"E:\car_search_data\eda\type_v2")
FAR_TARGET = 0.05


def threshold_for_far(unknown_scores: np.ndarray, far=FAR_TARGET) -> float:
    """Lowest threshold t such that the share of unknowns with score > t is <= far (score equal to t is refused)."""
    return float(np.sort(unknown_scores)[int(np.ceil(len(unknown_scores) * (1 - far))) - 1])


def rate(scores: np.ndarray, thr: float, answered: bool) -> dict:
    """Share answered (score > thr) or refused, with a bootstrap CI."""
    if not len(scores):
        return {"value": None, "n": 0}
    v = (scores > thr).astype(float) if answered else (scores <= thr).astype(float)
    lo, hi = bootstrap_ci(v)
    return {"value": round(float(v.mean()), 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4), "n": len(v)}


def auroc(pos: np.ndarray, neg: np.ndarray):
    from sklearn.metrics import roc_auc_score
    if not len(pos) or not len(neg):
        return None
    return round(float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])), 4)


def risk_coverage(score: np.ndarray, correct: np.ndarray, points=21) -> list:
    """Known queries only: coverage (share answered) against risk (top-1 error among the answered)."""
    out = []
    for q in np.linspace(0, 0.95, points):
        thr = float(np.quantile(score, q))
        keep = score > thr
        out.append({"coverage": round(float(keep.mean()), 4), "risk": round(float(1 - correct[keep].mean()), 4), "threshold": round(thr, 4)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default="frozen")
    ap.add_argument("--alphas", default="1.0", help="WiSE-FT alphas for a fine-tuned checkpoint (ignored for frozen)")
    ap.add_argument("--unknowns", type=Path, default=REPO / "data/processed/queries_unanswerable_v1.json")
    ap.add_argument("--eval-split", type=Path, default=EDA / "splits_type_v2_eval.json")
    ap.add_argument("--train-split", type=Path, default=EDA / "splits_type_v2_train.json")
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--smoke", action="store_true", help="plumbing check: ~100 images, numbers are meaningless")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import pandas as pd
    from PIL import Image
    from torchvision import transforms as T
    from transformers import CLIPVisionModelWithProjection

    held = set(json.loads(a.train_split.read_text(encoding="utf-8"))["heldout_part_ids"])
    ev = json.loads(a.eval_split.read_text(encoding="utf-8"))
    gallery = [r for r in ev["catalog"] if r["part_id"] not in held]  # held-out types are hidden: their queries are unknowns
    known = ev["queries"]["valid"]
    unk = json.loads(a.unknowns.read_text(encoding="utf-8"))["items"]
    if a.smoke:
        keep = sorted({r["part_id"] for r in known})[:4]
        gallery, known = [r for r in gallery if r["part_id"] in keep][:40], [r for r in known if r["part_id"] in keep][:20]
        unk = [u for k in ("near", "far") for s in ("valid", "test") for u in [x for x in unk if x["kind"] == k and x["split"] == s][:8]]
    print(f"gallery {len(gallery)} (held-out types hidden: {sorted(held)}), known queries {len(known)}, unknowns {len(unk)}", flush=True)

    dev = a.device
    model = CLIPVisionModelWithProjection.from_pretrained(a.model).to(dev).eval()
    proj = torch.nn.Linear(768, 768, bias=False).to(dev)
    frozen = {g: {k: v.to(dev) for k, v in sd.items()} for g, sd in frozen_state(model).items()}
    tuned = None if a.checkpoint == "frozen" else torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    alphas = [0.0] if tuned is None else [float(x) for x in a.alphas.split(",")]

    prep = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.PILToTensor()])
    paths = sorted({r["image"] for r in gallery + known} | {u["image"] for u in unk})
    cache = {}
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            cache[p] = prep(im.convert("RGB"))
        if i % 1000 == 0:
            print(f"decoded {i}/{len(paths)}", flush=True)
    mean = torch.tensor(CLIP_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=dev).view(1, 3, 1, 1)

    def load_state(state):
        with torch.no_grad():
            model.vision_model.load_state_dict({k: v.to(dev) for k, v in state["vision_model"].items()})
            model.visual_projection.load_state_dict({k: v.to(dev) for k, v in state["visual_projection"].items()})
            proj.load_state_dict({k: v.to(dev) for k, v in state["proj"].items()})

    def embed(ps):
        out = {}
        for i in range(0, len(ps), a.batch):
            chunk = ps[i:i + a.batch]
            x = (torch.stack([cache[p] for p in chunk]).to(dev).float() / 255 - mean) / std
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                f = model.visual_projection(model.vision_model(pixel_values=x).pooler_output)
                e = torch.nn.functional.normalize(proj(f.float()).float(), dim=-1).cpu().numpy()
            out.update(zip(chunk, e))
        return out

    a.out.mkdir(parents=True, exist_ok=True)
    for alpha in alphas:
        load_state(frozen if tuned is None or alpha == 0.0 else blend(frozen, tuned, alpha))
        E = embed(paths)
        G = np.stack([E[r["image"]] for r in gallery])
        g_types = [r["part_id"] for r in gallery]

        def score(rows, E=E, G=G):
            return (np.stack([E[r["image"]] for r in rows]) @ G.T).max(1) if rows else np.array([])

        s_known = score(known)
        m = type_metrics(np.stack([E[r["image"]] for r in known]) @ G.T, g_types, [r["part_id"] for r in known])
        correct = np.array([float(t == p) for t, p in zip([r["part_id"] for r in known], m["top1"])])
        sc = {(u["kind"], u["split"]): [] for u in unk}
        per = defaultdict(list)  # (split, type) -> unknown items
        for u in unk:
            sc[(u["kind"], u["split"])].append(u)
            per[(u["split"], u["type"] if u["kind"] == "near" else "motorcycle")].append(u)
        S = {k: score(v) for k, v in sc.items()}
        res = {"checkpoint": a.checkpoint, "alpha": alpha, "smoke": a.smoke, "far_target": FAR_TARGET,
               "gallery": len(gallery), "known_queries": len(known), "known_recall@1": round(float(correct.mean()), 4),
               "unknown_counts": {f"{k}_{s}": len(v) for (k, s), v in sc.items()}, "calibrations": {}}
        calib_scores = {"near": S.get(("near", "valid"), np.array([])),
                        "all": np.r_[S.get(("near", "valid"), np.array([])), S.get(("far", "valid"), np.array([]))]}
        for name, cs in calib_scores.items():
            if not len(cs):
                continue
            thr = threshold_for_far(cs)
            r = {"threshold": round(thr, 4), "chosen_on": f"validation unknowns ({name}), FAR <= {FAR_TARGET:.0%}",
                 "FAR_validation_near": rate(S.get(("near", "valid"), np.array([])), thr, True),
                 "FAR_validation_far": rate(S.get(("far", "valid"), np.array([])), thr, True),
                 "FAR_test_near": rate(S.get(("near", "test"), np.array([])), thr, True),
                 "FAR_test_far": rate(S.get(("far", "test"), np.array([])), thr, True),
                 "false_abstention_known": rate(s_known, thr, False),
                 "coverage_known": rate(s_known, thr, True),
                 "selective_accuracy_known": round(float(correct[s_known > thr].mean()), 4) if (s_known > thr).any() else None,
                 "FAR_test_per_type": {ty: rate(score(v), thr, True) for (sp, ty), v in sorted(per.items()) if sp == "test"}}
            res["calibrations"][name] = r
        res["auroc_known_vs_unknown"] = {f"{k}_{s}": auroc(s_known, v) for (k, s), v in S.items()}
        res["risk_coverage_known"] = risk_coverage(s_known, correct)
        sweep = []
        for t in np.linspace(float(min(s_known.min(), *[v.min() for v in S.values() if len(v)])),
                             float(max(s_known.max(), *[v.max() for v in S.values() if len(v)])), 101):
            sweep.append({"threshold": round(float(t), 4), "coverage_known": round(float((s_known > t).mean()), 4),
                          **{f"FAR_{k}_{s}": round(float((v > t).mean()), 4) for (k, s), v in S.items() if len(v)}})
        pd.DataFrame(sweep).to_csv(a.out / f"curve_a{alpha:.2f}.csv", index=False)
        (a.out / f"a6_a{alpha:.2f}.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        c = res["calibrations"].get("near", {})
        print(f"alpha {alpha}: known R@1 {res['known_recall@1']}, near-calibrated thr {c.get('threshold')}, "
              f"FAR test near {c.get('FAR_test_near', {}).get('value')} far {c.get('FAR_test_far', {}).get('value')}, "
              f"false abstention {c.get('false_abstention_known', {}).get('value')}", flush=True)
    print(f"done -> {a.out}")


if __name__ == "__main__":
    main()
