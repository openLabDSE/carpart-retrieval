#!/usr/bin/env bash
# Public sources for D1 (detection set). Run from the repo root.
# NONE of these supply part numbers - D2 (catalog) and D3 (query photos)
# must be built by hand. See docs/DATASET_SPEC.md.
set -euo pipefail
mkdir -p data/raw/roboflow

# --- Roboflow Universe -------------------------------------------------------
# Needs your own free API key: https://app.roboflow.com  -> Settings -> API
# Record licence + download date for EVERY dataset in data/raw/roboflow/SOURCES.md
#   pip install roboflow
#   export ROBOFLOW_API_KEY=...
# python - <<'PY'
# import os
# from roboflow import Roboflow
# rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
# (rf.workspace("project-p5nyc").project("car-parts-o7dlr")
#    .version(1).download("yolov8", location="data/raw/roboflow/car-parts-o7dlr"))
# PY

# --- Kaggle ------------------------------------------------------------------
#   pip install kaggle   # then put kaggle.json in ~/.kaggle/
# kaggle datasets download -d gpiosenka/car-parts-40-classes -p data/raw/roboflow --unzip
# kaggle datasets download -d qubdidata/auto-parts-dataset    -p data/raw/roboflow --unzip
#
# "50 Types of Car Parts" (~308 MB, 224x224 JPG, 50 part TYPE labels, no part
# numbers): D2 distractor source only, never queries. Slug confirmed 2026-10-14:
# kaggle datasets download -d gpiosenka/car-parts-40-classes -p data/raw/public/kaggle-car-parts-50 --unzip
# Licence: check the Kaggle page's licence box before use, record in
# data/raw/public/SOURCES.md. Before any use in fine-tuning: drop lighting and
# suspension/steering classes (RQ5 held-out categories, RESEARCH_PROTOCOL 3.2).
#
# --- CompCars (Yang et al., CVPR 2015) ---------------------------------------
# 136,727 whole-car + 27,618 car-part images, non-commercial research licence.
# No signed agreement/email needed - direct password-gated download
# (checked 2026-10-14, http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html):
#   Dropbox:      https://www.dropbox.com/sh/46de2cre37fvzu6/AABXtX8QqA6sx37k1IyZmNQ2a?dl=0
#   Google Drive: https://drive.google.com/open?id=18EunmjOJsbE5Lh9zA0cZ4wKV6Um46dkg
#   Password: d89551fd190e38  (grab data.zip + data.z01-z22, and sv_data.* for surveillance)
# mkdir -p data/raw/public/compcars
# zip -F data.zip --out combined.zip && unzip -P d89551fd190e38 combined.zip
# Usage terms are accepted by use (six-point non-commercial clause on the
# index page), not by a returned agreement. Cite Yang et al. CVPR 2015.
# Weak fit for this thesis: car parts are shown mounted on the car, not loose
# on a counter (out of scope, DATASET_SPEC.md S1). Use only as a detection
# pretrain source if at all, never as catalogue/query ground truth.
#
# --- Access by request (no public direct download, checked 2026-10-14) ------
# LSCP (arXiv 2311.11754, 84,162 images, 12 part types, detection): no dataset
#   link in the paper or on arXiv; email the authors (Wang Jie, Zhong Yilin,
#   Cao Qianqian) for access. Useful only for detector testing (W8), not
#   catalogue/query data - it has no part numbers.
# Go-CaRD / MuSe-CaR-Part + Close-CAR (arXiv 2006.08521): GPL-3.0 code at
#   https://github.com/lstappen/GoCarD, data on Zenodo
#   (zenodo.org/record/4450468 dataset, /record/4453520 models). Non-commercial
#   research only - download the EULA from the repo, sign it, and request
#   access before pulling anything. Useful as a detector/condition example
#   only, not for retrieval ground truth.
#
# --- Cite only, do not attempt to download -----------------------------------
# nyris Automotive v1 (nyris Visual Search Benchmark, benchmark.nyris.io):
#   private evaluation set built from a distributor's catalogue + user query
#   photos, not released publicly. Cite the benchmark paper/blog for numbers;
#   do not add a download step for it.

# --- Hugging Face ------------------------------------------------------------
#   pip install huggingface_hub
# huggingface-cli download DrBimmer/car-parts-and-damage-dataset \
#     --repo-type dataset --local-dir data/raw/roboflow/drbimmer

echo "Uncomment the source you want, add your key, and re-run."
echo "Then: python scripts/data/audit_dataset.py data/raw/roboflow"
