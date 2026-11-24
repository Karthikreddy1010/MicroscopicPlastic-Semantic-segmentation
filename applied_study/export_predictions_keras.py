"""
Export probability maps from the proposed Keras model (and, optionally, the Keras ablation
variants) in the layout expected by evaluate_quantification.py:

    PRED_ROOT/<Model>/seed<S>/{val,test,clean}/<stem>.png   (uint8 = probability * 255, 512x512)

HOW TO USE: paste this as the LAST cell of three_seed.ipynb or experiment_B_loss_function_study.ipynb,
after the cells that define IMG_HEIGHT/IMG_WIDTH, preprocess_input, the custom layers and the
build_efficientnetv2b3_* functions. Then fill WEIGHTS with the saved *.weights.h5 files
(from your Google Drive results folder) and run it. No retraining is needed.
"""
import os

import numpy as np
from PIL import Image

DATA_ROOT = "/content"   # folder that contains training/, val/, Testing/
PRED_ROOT = "preds"
# Folder that contains experiment_B_loss/ (as saved by experiment_B_loss_function_study.ipynb).
# If you copied it to Google Drive, mount Drive and point this there, e.g. "/content/drive/MyDrive/results".
RESULTS_ROOT = "results"

# (model name in the paper, builder, {seed: weights path})
WEIGHTS = [
    ("Proposed", build_efficientnetv2b3_mrfm_attention_unet, {   # noqa: F821 (defined in the notebook)
        s: os.path.join(RESULTS_ROOT, "experiment_B_loss", f"seed_{s}", f"seed_{s}_final.weights.h5")
        for s in (42, 123, 2024)   # Tversky+Focal+Dice runs (final architecture + final loss)
    }),
    # Optional: the architecture ablation variants, so they are scored with the same protocol
    # ("Ablation_Baseline_EffV2B3", build_efficientnetv2b3_baseline_unet, {42: "...", 123: "...", 2024: "..."}),
    # ("Ablation_MRFM", build_efficientnetv2b3_mrfm_unet, {...}),
    # ("Ablation_Attention", build_efficientnetv2b3_attention_unet, {...}),
    # ("Ablation_MRFM_Attention_twophase", build_efficientnetv2b3_mrfm_attention_unet, {...}),
]

VAL_DIR = os.path.join(DATA_ROOT, "val/val/Original")
TEST_DIR = os.path.join(DATA_ROOT, "Testing/Testing/Original")


def _load(path):
    img = Image.open(path).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)  # noqa: F821
    return np.asarray(img, dtype=np.float32) / 255.0


def _export(model, out_dir, img_dir, keep):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")) and keep(f))
    for i in range(0, len(files), 16):
        chunk = files[i:i + 16]
        x = preprocess_input(np.stack([_load(os.path.join(img_dir, f)) for f in chunk]) * 255.0)  # noqa: F821
        probs = model.predict(x, batch_size=4, verbose=0)[..., 0]
        for f, p in zip(chunk, probs):
            Image.fromarray(np.round(p * 255).astype(np.uint8)).save(
                os.path.join(out_dir, os.path.splitext(f)[0] + ".png"))
    return len(files)


_missing = [p for _, _, s in WEIGHTS for p in s.values() if not os.path.exists(p)]
if _missing:
    raise FileNotFoundError("Weights not found (set RESULTS_ROOT):\n  " + "\n  ".join(_missing))

for name, builder, seeds in WEIGHTS:
    for seed, wpath in seeds.items():
        model, _ = builder(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))  # noqa: F821
        model.load_weights(wpath)
        run = os.path.join(PRED_ROOT, name, f"seed{seed}")
        n_v = _export(model, os.path.join(run, "val"), VAL_DIR, lambda f: "Clean" not in f)
        n_t = _export(model, os.path.join(run, "test"), TEST_DIR, lambda f: "Clean" not in f)
        n_c = _export(model, os.path.join(run, "clean"), TEST_DIR, lambda f: "Clean" in f)
        print(f"{name} seed{seed}: {n_v} val, {n_t} test, {n_c} blank maps -> {run}  "
              f"({model.count_params() / 1e6:.1f}M params)")
        del model
        tf.keras.backend.clear_session()  # noqa: F821

# Zip and download PRED_ROOT, or copy it next to the baseline predictions:
# !zip -qr preds_keras.zip preds
