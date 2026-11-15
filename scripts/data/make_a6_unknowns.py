#!/usr/bin/env python3
"""A6 unknown-part set: near unknowns (car parts not in the catalogue) and far unknowns (motorcycle parts).

Plan: Experiments/A6 Unknown-Part Set Plan (2026-11-12). Built from data already held; nothing is downloaded.
- Option 1: the 100 held-out-type queries of type_v2 (air filter, camshaft, fuel injector, side mirror, transmission).
  The locked split files are not changed; A6 hides these types from the gallery at evaluation time.
- Option 2: types dropped from type_v2 (bulb, cabin air filter, timing belt, wiper blade) from Commons plus typelevel_v1
  (bulb, cabin_filter, review_decision=keep). Near-duplicates are removed within the group and against every
  type_v2 image (pHash <= 6 plus thumbnail check, same rule as the split scripts).
- Far unknowns: one photo per motorcycle part label (the same 539 as eval_a5.py --far-unknowns).
- Validation/test split by TYPE (decided 2026-11-12); far unknowns by label, half each, seed 42.
Writes data/processed/queries_unanswerable_v1.json and a report next to it.

Usage: python scripts/data/make_a6_unknowns.py
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_ccby_split import thumb
from make_type_split import near_groups

REPO = Path(__file__).resolve().parents[2]
E = Path(r"E:\car_search_data")
EDA = E / "eda/type_v2"
OUT = REPO / "data/processed/queries_unanswerable_v1.json"
VALID_TYPES = {"camshaft", "side mirror", "bulb", "timing belt"}
TEST_TYPES = {"air filter", "fuel injector", "transmission", "cabin air filter", "wiper blade"}
COMMONS_TYPES = {"bulb", "cabin air filter", "timing belt", "wiper blade"}
TYPELEVEL = {"bulb": "bulb", "cabin_filter": "cabin air filter"}
PRIORITY = {"commons": 0, "openverse": 1}


def hash_image(path):
    rgb = Image.open(path).convert("RGB")
    return int(str(imagehash.phash(rgb)), 16), thumb(rgb)


def main():
    rng = random.Random(42)
    train = json.loads((EDA / "splits_type_v2_train.json").read_text(encoding="utf-8"))
    ev = json.loads((EDA / "splits_type_v2_eval.json").read_text(encoding="utf-8"))
    held = ev["queries"]["heldout"]
    ref_paths = sorted({r["image"] for r in train["catalog"] + ev["catalog"] + ev["queries"]["valid"] + held})
    print(f"type_v2 reference images {len(ref_paths)}", flush=True)

    cand = []
    c = pd.read_csv(E / "dataset_clean/metadata/clean_images.csv", dtype=str).fillna("")
    for r in c[c["class"].isin(COMMONS_TYPES)].to_dict("records"):
        cand.append({"id": r["image_id"], "image": Path(r["clean_path"]).as_posix(), "type": r["class"],
                     "src": "commons", "licence": r["licence"], "source_url": r["source_url"]})
    t = pd.read_csv(E / "typelevel_v1/originals_index.csv", dtype=str, encoding="utf-8-sig").fillna("")
    for r in t[(t["type"].isin(TYPELEVEL)) & (t["review_decision"] == "keep")].to_dict("records"):
        cand.append({"id": r["id"], "image": Path(r["local_path"]).as_posix(), "type": TYPELEVEL[r["type"]],
                     "src": "openverse", "licence": r["license"], "source_url": r["source"]})
    print(f"option-2 candidates {len(cand)} {dict(Counter((x['type'], x['src']) for x in cand))}", flush=True)

    ph, ths = [], []
    for p in [x["image"] for x in cand] + ref_paths:
        h, th = hash_image(p)
        ph.append(h)
        ths.append(th)
    ph = np.array(ph, dtype=np.uint64)
    n = len(cand)
    uf = near_groups(ph, ths)
    groups = defaultdict(list)
    for i in range(len(ph)):
        groups[uf.find(i)].append(i)
    kept, dup_ref, dup_within, conflict = [], 0, 0, 0
    for members in groups.values():
        cm = [i for i in members if i < n]
        if not cm:
            continue
        if any(i >= n for i in members):
            dup_ref += len(cm)
            continue
        if len({cand[i]["type"] for i in cm}) > 1:
            conflict += len(cm)
            continue
        keep = min(cm, key=lambda i: (PRIORITY[cand[i]["src"]], cand[i]["image"]))
        kept.append(cand[keep])
        dup_within += len(cm) - 1
    print(f"kept {len(kept)}; dropped as copies of type_v2 {dup_ref}, within-group {dup_within}, label conflict {conflict}", flush=True)

    items = []
    for r in held:
        items.append({"id": Path(r["image"]).stem, "image": r["image"], "type": r["part_id"], "kind": "near",
                      "option": 1, "src": r["src"], "licence": "see type_v2 dataset card"})
    for r in sorted(kept, key=lambda x: x["id"]):
        items.append({"id": r["id"], "image": r["image"], "type": r["type"], "kind": "near", "option": 2,
                      "src": r["src"], "licence": r["licence"], "source_url": r["source_url"]})
    for it in items:
        it["split"] = "valid" if it["type"] in VALID_TYPES else "test" if it["type"] in TEST_TYPES else None
        assert it["split"], it

    m = pd.read_csv(E / "dataset_combined/metadata/combined_images.csv", dtype=str)
    far = m[m.label_type == "part"].sort_values("image_id").groupby("label").head(1)
    labels = sorted(far.label.unique())
    rng.shuffle(labels)
    half = set(labels[:len(labels) // 2])
    for r in far.to_dict("records"):
        items.append({"id": r["image_id"], "image": Path(r["clean_path"]).as_posix(), "type": "motorcycle part:" + r["label"],
                      "kind": "far", "option": 0, "src": "roboflow_combined", "licence": r["licence"],
                      "split": "valid" if r["label"] in half else "test"})

    cnt = Counter((i["split"], i["kind"], i["type"] if i["kind"] == "near" else "motorcycle") for i in items)
    note = ("Unanswerable queries for A6. Near = car part types not in the gallery during A6 (option 1: type_v2 held-out types, "
            "hidden from the gallery at evaluation time; option 2: types dropped from type_v2). Far = motorcycle parts. "
            "Split by type (near) and by label (far). Set the threshold on split=valid, report on split=test.")
    OUT.write_text(json.dumps({"seed": 42, "note": note, "items": items}, indent=1), encoding="utf-8")
    lines = ["# A6 unknown-part set (queries_unanswerable_v1)", "", note, "",
             f"- option-2 candidates {len(cand)}, kept {len(kept)}, copies of type_v2 {dup_ref}, within-group duplicates {dup_within}, label conflicts {conflict}",
             f"- total {len(items)}: near {sum(i['kind'] == 'near' for i in items)}, far {sum(i['kind'] == 'far' for i in items)}",
             "", "| split | kind | type | photos |", "|---|---|---|---|"]
    lines += [f"| {s} | {k} | {ty} | {v} |" for (s, k, ty), v in sorted(cnt.items())]
    OUT.with_suffix(".report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[8:]), flush=True)


if __name__ == "__main__":
    main()
