"""
Seed ensemble: average the probability maps of the seeds of each model (pixel-wise mean).
Writes PRED_ROOT_OUT/<Model>/seed0/{val,test,clean}/<stem>.png in the layout the evaluator expects,
so the ensemble is scored with exactly the same protocol as the single-seed models.

Usage:
    python make_ensemble.py --pred-root ../../preds --out ../../preds_ensemble
"""
import argparse
import os
import re

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    for model in sorted(os.listdir(args.pred_root)):
        mdir = os.path.join(args.pred_root, model)
        seeds = sorted(s for s in os.listdir(mdir) if re.fullmatch(r"seed\d+", s))
        if not seeds:
            continue
        for split in ("val", "test", "clean"):
            stems = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(mdir, seeds[0], split)))
            out_dir = os.path.join(args.out, model, "seed0", split)
            os.makedirs(out_dir, exist_ok=True)
            for stem in stems:
                acc = None
                for s in seeds:
                    im = np.asarray(Image.open(os.path.join(mdir, s, split, stem + ".png")).convert("L"), dtype=np.float32)
                    acc = im if acc is None else acc + im
                avg = np.round(acc / len(seeds)).astype(np.uint8)
                Image.fromarray(avg).save(os.path.join(out_dir, stem + ".png"))
        print(f"{model}: averaged {len(seeds)} seeds {seeds}")


if __name__ == "__main__":
    main()
