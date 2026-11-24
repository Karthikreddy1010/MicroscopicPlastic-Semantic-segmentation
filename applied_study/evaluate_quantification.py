"""
Framework-agnostic evaluation for the applied microplastic study.

Scores every model with the SAME protocol, from saved probability maps:
  * pixel metrics at 512x512 (same convention as the original pipeline)
  * boundary F1 (BF score) at native resolution
  * particle-level detection (precision / recall / F1) and per-image COUNT agreement
    (MAE, bias, Pearson r, Lin's CCC, Bland-Altman limits of agreement)
  * particle SIZE agreement (equivalent circular diameter) on TRACED particles only
  * false particles on BLANK (clean, particle-free) dishes
  * paired per-image statistics of every model against a reference model

Expected prediction layout (one PNG per image, uint8 = probability * 255, any size):
    PRED_ROOT/<Model>/seed<S>/val/<stem>.png
    PRED_ROOT/<Model>/seed<S>/test/<stem>.png
    PRED_ROOT/<Model>/seed<S>/clean/<stem>.png      (optional, blank dishes)
<stem> is the image filename without extension.

The binarisation threshold is chosen per model+seed on the VALIDATION predictions
(pooled Dice, grid 0.05..0.95 in steps of 0.05) and then frozen for test and blanks - identical for all models.

Annotation note: small particles in this dataset are labelled with a fixed ~20 px disk
(area ~310-345 px, bbox 19-22 px) instead of a traced outline. These "stamp" particles are
valid for COUNTING but not for SIZE, so size statistics use traced particles only.

Usage:
    python evaluate_quantification.py --data-root ../Microscopic_plastic_dataset \
        --pred-root preds --out results_applied --reference "Proposed" [--um-per-px 4.2]
"""
import argparse
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

IMG = 512
STRUCT = np.ones((3, 3), dtype=bool)  # 8-connectivity
SPLITS = {
    "val": ("val/val/Original", "val/val/val_100masks"),
    "test": ("Testing/Testing/Original", "Testing/Testing/testing_100masks"),
}
CLEAN_DIR = "Testing/Testing/Original"  # x-Testing-Clean-*.jpg are blank dishes


# ----------------------------------------------------------------------------- IO
def load_gt(data_root, split):
    """Return {stem: native bool mask} for every annotated image of a split."""
    img_dir, mask_dir = (os.path.join(data_root, p) for p in SPLITS[split])
    out = {}
    for f in sorted(os.listdir(mask_dir)):
        stem = f[: -len("_mask.png")]
        if not os.path.exists(os.path.join(img_dir, stem + ".jpg")):
            continue
        out[stem] = np.array(Image.open(os.path.join(mask_dir, f)).convert("L")) > 127
    return out


def clean_stems(data_root):
    d = os.path.join(data_root, CLEAN_DIR)
    return {os.path.splitext(f)[0]: Image.open(os.path.join(d, f)).size[::-1]
            for f in sorted(os.listdir(d)) if "Clean" in f and "Copy" not in f}


def load_prob(path, shape):
    """Load a probability PNG and resize (bilinear) to `shape` (H, W); returns float [0,1]."""
    im = Image.open(path).convert("L")
    if im.size != (shape[1], shape[0]):
        im = im.resize((shape[1], shape[0]), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def to512(mask_bool):
    """GT preprocessing identical to the training pipeline (nearest resize, >127)."""
    im = Image.fromarray(mask_bool.astype(np.uint8) * 255).resize((IMG, IMG), Image.NEAREST)
    return np.asarray(im) > 127


# ------------------------------------------------------------------------ metrics
def pooled_pixel(gts, preds):
    tp = fp = fn = tn = 0
    for g, p in zip(gts, preds):
        tp += np.sum(g & p); fp += np.sum(~g & p); fn += np.sum(g & ~p); tn += np.sum(~g & ~p)
    fg = tp / max(tp + fp + fn, 1)
    bg = tn / max(tn + fp + fn, 1)
    return dict(Dice=2 * tp / max(2 * tp + fp + fn, 1), FG_IoU=fg, mIoU=(fg + bg) / 2,
                Precision=tp / max(tp + fp, 1), Recall=tp / max(tp + fn, 1))


def boundary_f1(g, p, tol=3):
    """BF score: boundary pixels matched within `tol` px (Csurka et al., 2013)."""
    bg = g & ~ndi.binary_erosion(g, STRUCT)
    bp = p & ~ndi.binary_erosion(p, STRUCT)
    if bg.sum() == 0 and bp.sum() == 0:
        return 1.0
    if bg.sum() == 0 or bp.sum() == 0:
        return 0.0
    dg = ndi.distance_transform_edt(~bg)
    dp = ndi.distance_transform_edt(~bp)
    prec = np.mean(dg[bp] <= tol)
    rec = np.mean(dp[bg] <= tol)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def split_labels(mask, min_dist):
    """Label touching particles separately: watershed on the distance transform, one marker per
    distance-map peak (peaks at least `min_dist` px apart). Every connected component keeps at
    least one marker, so no predicted pixel is lost."""
    lab0, _ = ndi.label(mask, structure=STRUCT)
    dist = ndi.distance_transform_edt(mask)
    coords = peak_local_max(dist, min_distance=min_dist, labels=lab0, exclude_border=False)
    markers = np.zeros(mask.shape, dtype=np.int32)
    k = 0
    for r, c in coords:
        k += 1; markers[r, c] = k
    hit = set(lab0[markers > 0].tolist())
    for comp in range(1, lab0.max() + 1):
        if comp not in hit:
            r, c = np.unravel_index(np.argmax(np.where(lab0 == comp, dist, -1)), mask.shape)
            k += 1; markers[r, c] = k
    return watershed(-dist, markers, mask=mask)


def components(mask, min_area=0, split_dist=0):
    lab = split_labels(mask, split_dist) if split_dist else ndi.label(mask, structure=STRUCT)[0]
    objs = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        m = lab[sl] == i
        a = int(m.sum())
        if a < min_area:
            lab[sl][m] = 0
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        cy, cx = ndi.center_of_mass(m)
        objs.append(dict(id=i, area=a, ecd=2 * np.sqrt(a / np.pi),
                         cy=cy + sl[0].start, cx=cx + sl[1].start,
                         stamp=(300 <= a <= 345 and 19 <= h <= 22 and 19 <= w <= 22)))
    return lab, objs


def match_objects(lab_g, objs_g, lab_p, objs_p, dilate=3):
    """One-to-one matching: a predicted particle may match a GT particle only if it overlaps
    the GT particle (GT dilated by `dilate` px); among eligible pairs the Hungarian
    algorithm minimises centroid distance. Overlap-based eligibility (not IoU>=0.5) is used
    because stamped GT disks are larger than the particles they mark."""
    if not objs_g or not objs_p:
        return []
    gid = {o["id"]: k for k, o in enumerate(objs_g)}
    pid = {o["id"]: k for k, o in enumerate(objs_p)}
    elig = np.zeros((len(objs_g), len(objs_p)), dtype=bool)
    for o in objs_g:
        region = ndi.binary_dilation(lab_g == o["id"], STRUCT, iterations=dilate)
        for v in np.unique(lab_p[region]):
            if v in pid:
                elig[gid[o["id"]], pid[v]] = True
    gc = np.array([[o["cy"], o["cx"]] for o in objs_g])
    pc = np.array([[o["cy"], o["cx"]] for o in objs_p])
    cost = np.linalg.norm(gc[:, None] - pc[None], axis=2)
    cost[~elig] = 1e9
    r, c = linear_sum_assignment(cost)
    return [(i, j) for i, j in zip(r, c) if elig[i, j]]


def lins_ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return np.nan
    cov = np.mean((x - x.mean()) * (y - y.mean()))
    return 2 * cov / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)


def choose_threshold(val_gt512, val_probs512):
    best_t, best_d = 0.5, -1
    for t in np.round(np.arange(0.05, 0.951, 0.05), 2):
        d = pooled_pixel(val_gt512, [p > t for p in val_probs512])["Dice"]
        if d > best_d:
            best_t, best_d = float(t), d
    return best_t, best_d


# ------------------------------------------------------------------------ per run
def evaluate_run(run_dir, gt_val, gt_test, blanks, args):
    vdir, tdir = os.path.join(run_dir, "val"), os.path.join(run_dir, "test")
    missing = [s for s in gt_test if not os.path.exists(os.path.join(tdir, s + ".png"))]
    if missing:
        raise FileNotFoundError(f"{run_dir}: {len(missing)} test predictions missing, e.g. {missing[0]}")
    vstems = [s for s in gt_val if os.path.exists(os.path.join(vdir, s + ".png"))]
    if len(vstems) < len(gt_val):
        warnings.warn(f"{run_dir}: only {len(vstems)}/{len(gt_val)} val predictions found")
    t, vdice = choose_threshold([to512(gt_val[s]) for s in vstems],
                                [load_prob(os.path.join(vdir, s + ".png"), (IMG, IMG)) for s in vstems])

    rows, g512, p512 = [], [], []
    gt_ecd, pr_ecd = [], []
    for s, g in gt_test.items():
        path = os.path.join(tdir, s + ".png")
        g512.append(to512(g)); p512.append(load_prob(path, (IMG, IMG)) > t)
        p = load_prob(path, g.shape) > t  # native resolution for object / size / boundary
        lab_g, og = components(g)
        lab_p, op = components(p, args.min_area, args.split_dist)
        m = match_objects(lab_g, og, lab_p, op)
        tp = len(m)
        for i, j in m:
            if not og[i]["stamp"]:
                gt_ecd.append(og[i]["ecd"]); pr_ecd.append(op[j]["ecd"])
        inter = np.sum(g512[-1] & p512[-1]); union = np.sum(g512[-1] | p512[-1])
        rows.append(dict(stem=s, n_gt=len(og), n_pred=len(op), tp=tp, fp=len(op) - tp, fn=len(og) - tp,
                         n_gt_stamp=sum(o["stamp"] for o in og),
                         fg_iou_img=inter / union if union else 1.0, bf=boundary_f1(g, p, args.bf_tol)))
    per_img = pd.DataFrame(rows)

    blank_rows = []
    cdir = os.path.join(run_dir, "clean")
    if os.path.isdir(cdir):
        for s, shape in blanks.items():
            fp_path = os.path.join(cdir, s + ".png")
            if os.path.exists(fp_path):
                p = load_prob(fp_path, shape) > t
                _, op = components(p, args.min_area, args.split_dist)
                blank_rows.append(dict(n_false=len(op), area_frac=sum(o["area"] for o in op) / p.size))
    blanks_df = pd.DataFrame(blank_rows)

    TP, FP, FN = per_img.tp.sum(), per_img.fp.sum(), per_img.fn.sum()
    diff = per_img.n_pred - per_img.n_gt
    s = args.um_per_px or 1.0
    gt_ecd, pr_ecd = np.array(gt_ecd) * s, np.array(pr_ecd) * s
    res = dict(Threshold=t, Val_Dice=vdice, **pooled_pixel(g512, p512),
               BF_score=per_img.bf.mean(),
               Obj_Precision=TP / max(TP + FP, 1), Obj_Recall=TP / max(TP + FN, 1),
               Obj_F1=2 * TP / max(2 * TP + FP + FN, 1),
               Count_MAE=np.abs(diff).mean(), Count_Bias=diff.mean(),
               Count_LoA_low=diff.mean() - 1.96 * diff.std(ddof=1),
               Count_LoA_high=diff.mean() + 1.96 * diff.std(ddof=1),
               Count_Pearson=stats.pearsonr(per_img.n_gt, per_img.n_pred)[0] if per_img.n_pred.std() > 0 else np.nan,
               Count_CCC=lins_ccc(per_img.n_gt, per_img.n_pred),
               Total_GT=int(per_img.n_gt.sum()), Total_Pred=int(per_img.n_pred.sum()),
               Size_n=len(gt_ecd),
               Size_MAE=np.abs(pr_ecd - gt_ecd).mean() if len(gt_ecd) else np.nan,
               Size_MRE=np.mean(np.abs(pr_ecd - gt_ecd) / gt_ecd) if len(gt_ecd) else np.nan,
               Size_Pearson=stats.pearsonr(gt_ecd, pr_ecd)[0] if len(gt_ecd) > 2 else np.nan,
               Size_KS_p=stats.ks_2samp(gt_ecd, pr_ecd).pvalue if len(gt_ecd) > 2 else np.nan)
    if len(blanks_df):
        res.update(Blank_n=len(blanks_df), Blank_FalseParticles_mean=blanks_df.n_false.mean(),
                   Blank_pct_with_FP=100 * (blanks_df.n_false > 0).mean(),
                   Blank_FP_area_pct=100 * blanks_df.area_frac.mean())
    return res, per_img, (gt_ecd, pr_ecd)


# ---------------------------------------------------------------------- reporting
def bootstrap_ci(per_img, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(per_img))
    vals = [fn(per_img.iloc[rng.choice(idx, len(idx))]) for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])


def obj_f1(df):
    return 2 * df.tp.sum() / max(2 * df.tp.sum() + df.fp.sum() + df.fn.sum(), 1)


def make_figures(out, per_img_ref, sizes_ref, ref, unit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g, p = per_img_ref.n_gt, per_img_ref.n_pred
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    mx = max(g.max(), p.max()) + 1
    ax[0].scatter(g, p, s=18, alpha=.7); ax[0].plot([0, mx], [0, mx], "k--", lw=1)
    ax[0].set(xlabel="Annotated particles per image", ylabel="Predicted particles per image", title=f"{ref}: counts")
    mean, d = (g + p) / 2, p - g
    ax[1].scatter(mean, d, s=18, alpha=.7)
    for v, ls in [(d.mean(), "-"), (d.mean() + 1.96 * d.std(ddof=1), "--"), (d.mean() - 1.96 * d.std(ddof=1), "--")]:
        ax[1].axhline(v, color="k", ls=ls, lw=1)
    ax[1].set(xlabel="Mean of annotated and predicted count", ylabel="Predicted - annotated", title="Bland-Altman")
    ge, pe = sizes_ref
    if len(ge):
        bins = np.linspace(0, np.percentile(np.r_[ge, pe], 99), 25)
        ax[2].hist(ge, bins, alpha=.6, label="Annotated"); ax[2].hist(pe, bins, alpha=.6, label="Predicted")
        ax[2].legend()
    ax[2].set(xlabel=f"Equivalent circular diameter ({unit})", ylabel="Particles", title="Size, traced particles")
    safe = re.sub(r"\W+", "_", ref)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"quantification_{safe}.png"), dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--out", default="results_applied")
    ap.add_argument("--reference", default=None, help="model folder name used for paired tests and figures")
    ap.add_argument("--min-area", type=int, default=20, help="drop predicted particles smaller than this (native px)")
    ap.add_argument("--split-dist", type=int, default=0,
                    help="0 = off; else watershed-split touching particles, peaks >= this many px apart")
    ap.add_argument("--bf-tol", type=int, default=3)
    ap.add_argument("--um-per-px", type=float, default=None, help="physical scale; sizes reported in px if omitted")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gt_val, gt_test = load_gt(args.data_root, "val"), load_gt(args.data_root, "test")
    blanks = clean_stems(args.data_root)
    print(f"GT: {len(gt_val)} val, {len(gt_test)} test annotated; {len(blanks)} blank dishes")

    runs, per_img_all, sizes = [], {}, {}
    for model in sorted(os.listdir(args.pred_root)):
        mdir = os.path.join(args.pred_root, model)
        if not os.path.isdir(mdir):
            continue
        for sd in sorted(os.listdir(mdir)):
            if not re.fullmatch(r"seed\d+", sd):
                continue
            res, pim, sz = evaluate_run(os.path.join(mdir, sd), gt_val, gt_test, blanks, args)
            runs.append(dict(Model=model, Seed=int(sd[4:]), **res))
            per_img_all[(model, sd)] = pim.assign(Model=model, Seed=sd)
            sizes.setdefault(model, ([], []))
            sizes[model][0].extend(sz[0]); sizes[model][1].extend(sz[1])
            print(f"{model:30s} {sd:9s} t={res['Threshold']:.2f} FG_IoU={res['FG_IoU']:.4f} "
                  f"ObjF1={res['Obj_F1']:.3f} CountMAE={res['Count_MAE']:.2f}")
    if not runs:
        raise SystemExit("No runs found under --pred-root")

    by_seed = pd.DataFrame(runs)
    by_seed.to_csv(os.path.join(args.out, "summary_by_seed.csv"), index=False)
    num = by_seed.drop(columns=["Seed"]).groupby("Model")
    summary = num.mean().add_suffix("_mean").join(num.std().add_suffix("_sd"))
    summary = summary[[c for k in by_seed.columns[2:] for c in (k + "_mean", k + "_sd")]]
    summary.to_csv(os.path.join(args.out, "summary_mean_sd.csv"))
    per_img = pd.concat(per_img_all.values())
    per_img.to_csv(os.path.join(args.out, "per_image.csv"), index=False)

    # seed-averaged per-image values for paired tests / bootstrap
    agg = per_img.assign(abs_err=(per_img.n_pred - per_img.n_gt).abs()).groupby(["Model", "stem"]).mean(numeric_only=True)
    ci_rows = []
    for model in agg.index.get_level_values(0).unique():
        sub = per_img[per_img.Model == model]
        lo, hi = bootstrap_ci(sub, obj_f1)
        a = agg.loc[model]
        lo2, hi2 = bootstrap_ci(a.reset_index(), lambda d: d.abs_err.mean())
        ci_rows.append(dict(Model=model, Obj_F1_CI95=f"[{lo:.3f}, {hi:.3f}]",
                            Count_MAE_CI95=f"[{lo2:.2f}, {hi2:.2f}]"))
    pd.DataFrame(ci_rows).to_csv(os.path.join(args.out, "bootstrap_ci.csv"), index=False)

    if args.reference:
        if args.reference not in agg.index.get_level_values(0):
            raise SystemExit(f"--reference '{args.reference}' not found")
        ref = agg.loc[args.reference]
        rows = []
        for model in agg.index.get_level_values(0).unique():
            if model == args.reference:
                continue
            o = agg.loc[model].reindex(ref.index)
            row = dict(Model=model)
            for col, name in [("fg_iou_img", "FG_IoU_per_image"), ("abs_err", "Abs_count_error"), ("bf", "BF_score")]:
                d = ref[col] - o[col]
                p = stats.wilcoxon(ref[col], o[col]).pvalue if np.any(d != 0) else 1.0
                row[f"{name}_ref_minus_model"] = d.mean(); row[f"{name}_wilcoxon_p"] = p
            rows.append(row)
        st = pd.DataFrame(rows)
        if len(st):  # Holm correction across model comparisons, per metric
            for c in [c for c in st.columns if c.endswith("_p")]:
                order = np.argsort(st[c].values); m = len(st); adj = np.empty(m); run = 0
                for k, i in enumerate(order):
                    run = max(run, min(1, (m - k) * st[c].values[i])); adj[i] = run
                st[c + "_holm"] = adj
        st.to_csv(os.path.join(args.out, "paired_tests_vs_reference.csv"), index=False)
        seed0 = sorted(k for k in per_img_all if k[0] == args.reference)[0]
        make_figures(args.out, per_img_all[seed0], tuple(map(np.array, sizes[args.reference])),
                     args.reference, "um" if args.um_per_px else "px")

    with open(os.path.join(args.out, "protocol.json"), "w") as f:
        json.dump(vars(args) | dict(n_val=len(gt_val), n_test=len(gt_test), n_blank=len(blanks)), f, indent=2)
    pd.set_option("display.width", 200)
    cols = ["FG_IoU", "Dice", "BF_score", "Obj_F1", "Count_MAE", "Count_CCC", "Size_MRE", "Blank_FalseParticles_mean"]
    print("\n", by_seed.groupby("Model")[[c for c in cols if c in by_seed]].agg(["mean", "std"]).round(4))
    print(f"\nWrote results to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
