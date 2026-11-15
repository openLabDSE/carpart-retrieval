#!/usr/bin/env python3
"""A5 seeds: train and evaluate B (ft) and C (lpft) for seeds 42, 43, 44, plus the frozen arm A.

By default waits for hp_search_a5.py, reads the `WINNERS:` line from the search log, trains 10 epochs on the
tuning-train split with those settings, and evaluates each checkpoint on every set with the WiSE-FT alpha sweep
(D and E are the alpha sweeps of B and C). If the winners equal the config defaults, seed 42 of the original
type_v1 run is not re-run: the existing a5v2 checkpoints and results are reused. Finished runs are skipped.

--config picks the splits (its `train` section for training, its `data` section for evaluation); --tag names the
checkpoint and results folders; --winners skips the search log (e.g. to re-run tuned settings on a new split).

Usage:
  python scripts/run_a5_seeds.py
  python scripts/run_a5_seeds.py --config configs/a5/a5_type_v2.yaml --tag a5seeds_v2 --winners 0.05,1e-5,1e-3
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
LOG = Path("E:/car_search_data/results/hp_search.log")
ROOT = Path("E:/car_search_data")
DEFAULTS = (0.05, 3e-5, 1e-3)


def winners():
    while True:
        text = LOG.read_text(errors="replace") if LOG.exists() else ""
        m = re.search(r"WINNERS: temperature ([\d.e-]+), lr_ft ([\d.e-]+), lr_proj ([\d.e-]+)", text)
        if m:
            return tuple(float(x) for x in m.groups())
        if "Traceback" in text:
            sys.exit("hp search failed, see " + str(LOG))
        time.sleep(60)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=REPO / "configs/a5/a5.yaml")
    ap.add_argument("--tag", default="a5seeds", help="checkpoint and results folder name under E:/car_search_data")
    ap.add_argument("--winners", help="temperature,lr_ft,lr_proj; default: read from the hp search log")
    a = ap.parse_args()
    os.environ.setdefault("USE_TF", "0")
    temp, lr_ft, lr_proj = tuple(float(x) for x in a.winners.split(",")) if a.winners else winners()
    print("winners", temp, lr_ft, lr_proj, flush=True)
    ckpt_root, res_root = ROOT / "checkpoints" / a.tag, ROOT / "results" / a.tag
    data = yaml.safe_load(a.config.read_text(encoding="utf-8")).get("data", {})
    split_args = []  # eval_a5.py falls back to the type_v1 splits when the config does not name them
    for flag, key in (("--eval-split", "eval_split"), ("--train-split", "train_split"), ("--tune-split", "tune_eval_split")):
        if key in data:
            split_args += [flag, data[key]]
    evaluate = [sys.executable, str(REPO / "scripts/eval_a5.py"), *split_args]

    frozen = res_root / "A_frozen"
    if not list(frozen.glob("metrics_*.json")):
        subprocess.run([*evaluate, "--checkpoint", "frozen", "--out", str(frozen)], check=True, cwd=REPO)
    print("done A_frozen", flush=True)

    for arm, mode in (("B", "ft"), ("C", "lpft")):
        for seed in (42, 43, 44):
            if a.tag == "a5seeds" and seed == 42 and (temp, lr_ft, lr_proj) == DEFAULTS:
                print(f"{arm} seed 42: reusing a5v2", flush=True)
                continue
            name = f"{arm}_s{seed}"
            ckpt, res = ckpt_root / name, res_root / name
            if not (res / "metrics_a1.00.json").exists():
                cmd = [sys.executable, str(REPO / "scripts/train_embedding.py"), "--config", str(a.config),
                       "--mode", mode, "--epochs", "10", "--seed", str(seed), "--temperature", str(temp),
                       "--lr-ft", str(lr_ft), "--lr-proj", str(lr_proj), "--lr-lpft-ft", str(lr_ft / 10), "--out", str(ckpt)]
                subprocess.run(cmd, check=True, cwd=REPO)
                subprocess.run([*evaluate, "--checkpoint", str(ckpt / "model.pt"), "--alphas", "0,0.25,0.5,0.75,1",
                                "--out", str(res)], check=True, cwd=REPO)
            print("done", name, flush=True)


if __name__ == "__main__":
    main()
