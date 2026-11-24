# Applied study: baselines, particle counting, size and blank-dish evaluation

Three files, run in this order:

| Step | File | Where | What it does |
|---|---|---|---|
| 1 | `baselines_smp.ipynb` | Kaggle/Colab GPU (T4) | Trains 7 standard models x 3 seeds (42, 123, 2024) with the same split, augmentation, two-phase schedule and Tversky+Focal+Dice loss as the proposed model. Saves probability maps to `preds/<Model>/seed<S>/{val,test,clean}`. Resumable: finished runs are skipped. |
| 2 | `export_predictions_keras.py` | last cell of your Keras notebook | Loads your saved `.weights.h5` (no retraining) and exports the proposed model's maps into `preds/Proposed/seed<S>/...`. |
| 3 | `evaluate_quantification.py` | any CPU | Scores every model with one protocol and writes the tables, paired tests and figures. |

```bash
python evaluate_quantification.py --data-root ../Microscopic_plastic_dataset --pred-root preds --out results_applied --reference Proposed
```

Add `--um-per-px <value>` to report sizes in micrometres. For example, measure the inner dish
diameter in pixels and divide the known dish diameter (in µm) by it.

## Evaluation protocol (identical for every model)

- **Threshold:** chosen on the validation predictions (pooled Dice, grid 0.05 to 0.95), then frozen for test and blanks.
- **Pixel metrics:** computed at 512x512 with the original GT preprocessing (nearest resize, >127): Dice, FG IoU, mIoU, precision, recall.
- **Object metrics** (native resolution, 8-connected particles, predicted particles < 20 px removed):
  - A prediction can match a GT particle only if it overlaps that particle (dilated by 3 px).
  - Hungarian matching on centroid distance then gives object precision, recall and F1.
  - Per-image counts give count MAE, bias, Pearson r, Lin's CCC and Bland–Altman limits of agreement.
- **Size:** equivalent circular diameter of matched particles, on **traced** GT particles only (see the annotation note below).
- **Blanks:** the 100 `x-Testing-Clean-*` images are particle-free dishes. Every particle predicted on them is a false particle.
- **Statistics:** per-image values are averaged over seeds.
  - Paired Wilcoxon tests against `--reference`, Holm-corrected.
  - Bootstrap 95% CIs (2000 resamples over images).

## Dataset facts the paper must state

- **Annotated images:** train 556; val 100 (of 208 images; the rest are 104 blank `x-Clean` dishes and 4 unannotated `f--` images); test 100 (+100 blank dishes).
- **No leakage found:** no shared source names across splits, and no near-duplicate images between train and test (dHash).
- **Annotation style:** 41% of train particles and 56% of test particles are fixed ~20 px disks (area 310–345 px) marking small particles, not traced outlines. They are valid for counting but not for size measurement, and they put a ceiling on pixel IoU.
- **Particles per image:** train mean 7.9, test mean 4.9. The foreground fraction also differs (4.37% vs 1.79%), so the test set is sparser than training.

## Released outputs (manuscript submission)

| Folder / file | Contents |
|---|---|
| `preds/<Model>/seed<S>/{val,test,clean}/` | Probability maps (uint8 = probability × 255) for the proposed model and four baselines, 3 seeds each |
| `results_applied/` | Main results: per-seed and mean ± SD summaries, per-image results, bootstrap CIs, paired tests, error breakdown and error examples |
| `tune_postprocessing/` | Validation-only post-processing sweep (sensitivity analysis) |
| `results_ensemble/` | Seed-ensemble results (sensitivity analysis) |
| `Proposed_retrain_all_seeds.ipynb` | Notebook used to retrain the proposed model (seeds 42, 123, 2024) and export its predictions |

Reproduce the main results with:

    python evaluate_quantification.py --data-root ../Microscopic_plastic_dataset --pred-root preds --out results_applied --reference Proposed
