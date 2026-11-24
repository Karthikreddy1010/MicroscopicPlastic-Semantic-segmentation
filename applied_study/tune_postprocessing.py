"""
Choose the particle post-processing (minimum particle area, watershed split distance) on the
VALIDATION split only, using one shared setting for every model. The test split and the blank
dishes are not used here, so the reported test numbers stay untouched by this choice.

For each model+seed: the binarisation threshold is chosen on validation pixels (same rule as
evaluate_quantification.py), then object F1 is computed on validation for every
(min_area, split_dist) pair. The pair with the best mean validation object F1 across all runs
is printed and saved to tune_postprocessing/grid.csv.

Usage:
    python tune_postprocessing.py --data-root ../Microscopic_plastic_dataset --pred-root ../../preds
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

import evaluate_quantification as eq

MIN_AREAS = [0, 20, 40, 60, 80, 120]
SPLIT_DISTS = [0, 4, 6, 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--out", default="tune_postprocessing")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gt_val = eq.load_gt(args.data_root, "val")
    gt_lab = {s: eq.components(g) for s, g in gt_val.items()}  # GT objects, computed once
    gt512 = {s: eq.to512(g) for s, g in gt_val.items()}

    rows = []
    for model in sorted(os.listdir(args.pred_root)):
        mdir = os.path.join(args.pred_root, model)
        if not os.path.isdir(mdir):
            continue
        for sd in sorted(os.listdir(mdir)):
            if not re.fullmatch(r"seed\d+", sd):
                continue
            vdir = os.path.join(mdir, sd, "val")
            stems = [s for s in gt_val if os.path.exists(os.path.join(vdir, s + ".png"))]
            t, _ = eq.choose_threshold([gt512[s] for s in stems],
                                       [eq.load_prob(os.path.join(vdir, s + ".png"), (eq.IMG, eq.IMG)) for s in stems])
            probs = {s: eq.load_prob(os.path.join(vdir, s + ".png"), gt_val[s].shape) for s in stems}
            for split in SPLIT_DISTS:
                for ma in MIN_AREAS:
                    TP = FP = FN = 0
                    for s in stems:
                        lab_g, og = gt_lab[s]
                        lab_p, op = eq.components(probs[s] > t, ma, split)
                        m = len(eq.match_objects(lab_g, og, lab_p, op))
                        TP += m; FP += len(op) - m; FN += len(og) - m
                    rows.append(dict(Model=model, Seed=sd, Threshold=t, split_dist=split, min_area=ma,
                                     Obj_Precision=TP / max(TP + FP, 1), Obj_Recall=TP / max(TP + FN, 1),
                                     Obj_F1=2 * TP / max(2 * TP + FP + FN, 1)))
                print(f"{model:24s} {sd:8s} split={split} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "grid_by_run.csv"), index=False)
    mean = df.groupby(["split_dist", "min_area"])[["Obj_F1", "Obj_Precision", "Obj_Recall"]].mean()
    mean = mean.sort_values("Obj_F1", ascending=False)
    mean.to_csv(os.path.join(args.out, "grid.csv"))
    print("\nValidation object metrics, mean over all model-seed runs (best first):")
    print(mean.round(4).to_string())
    best = mean.index[0]
    print(f"\nBEST: --split-dist {best[0]} --min-area {best[1]}")


if __name__ == "__main__":
    main()
