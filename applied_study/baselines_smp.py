# %% [markdown]
# # Baseline segmentation models: same data, augmentation, loss and threshold protocol
#
# Trains standard segmentation architectures (segmentation_models_pytorch) on the exact split,
# preprocessing, augmentation and two-phase schedule used for the proposed model, for 3 seeds,
# and saves probability maps for val / test / blank dishes so that every model (including the
# Keras proposed model, exported with `export_predictions_keras.py`) is scored by
# `evaluate_quantification.py` with one identical protocol.
#
# Runs on a Kaggle / Colab T4. It is resumable: a (model, seed) whose predictions already exist
# is skipped, so you can split the work over several sessions.

# %%
# !pip install -q segmentation-models-pytorch timm scikit-image

# %%
import gc
import json
import os
import random
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from PIL import Image
from skimage.transform import resize, rotate

# ----------------------------------------------------------------------------- CONFIG
DATA_ROOT = os.environ.get("MP_DATA_ROOT", "/kaggle/input/microplastic/Microscopic_plastic_dataset")
PRED_ROOT = os.environ.get("MP_PRED_ROOT", "preds")      # PRED_ROOT/<Model>/seed<S>/{val,test,clean}
SEEDS = [42, 123, 2024]
IMG = 512
BATCH_SIZE = 4
PHASE1_EPOCHS, TOTAL_EPOCHS, PATIENCE = 40, 100, 15        # identical to the proposed model
LOSS_MODE = "tfd"   # "tfd" = Tversky+Focal+Dice (final proposed loss) | "two_phase" = baseline schedule
SMOKE = bool(int(os.environ.get("MP_SMOKE", "0")))         # tiny CPU run to check the code path

ENC = "tu-tf_efficientnetv2_b3"   # same backbone family as the proposed model (ImageNet weights)
MODELS = {
    "UNet_EffV2B3":       lambda: smp.Unet(ENC, encoder_weights="imagenet", classes=1),
    "UNetPP_EffV2B3":     lambda: smp.UnetPlusPlus(ENC, encoder_weights="imagenet", classes=1),
    "DeepLabV3P_EffV2B3": lambda: smp.DeepLabV3Plus(ENC, encoder_weights="imagenet", classes=1),
    "FPN_EffV2B3":        lambda: smp.FPN(ENC, encoder_weights="imagenet", classes=1),
    "MAnet_EffV2B3":      lambda: smp.MAnet(ENC, encoder_weights="imagenet", classes=1),
    "UNet_ResNet34":      lambda: smp.Unet("resnet34", encoder_weights="imagenet", classes=1),
    "SegFormer_MiTB2":    lambda: smp.Segformer("mit_b2", encoder_weights="imagenet", classes=1),
}
SESSION_MODELS = None   # e.g. ["UNet_ResNet34", "DeepLabV3P_EffV2B3"] to train only these
if SESSION_MODELS:
    MODELS = {k: MODELS[k] for k in SESSION_MODELS}


def find_data_root(root):
    """Use `root` if it exists, otherwise search /kaggle/input and /content for the dataset."""
    if os.path.isdir(os.path.join(root, "training")):
        return root
    import glob
    for base in ["/kaggle/input", "/content"]:
        hits = sorted(glob.glob(os.path.join(base, "**", "training", "training", "Training_original"), recursive=True))
        if hits:
            return os.path.dirname(os.path.dirname(os.path.dirname(hits[0])))
    raise FileNotFoundError(f"Dataset not found at {root}; set DATA_ROOT to the folder containing training/, val/, Testing/")


if SMOKE:
    SEEDS, PHASE1_EPOCHS, TOTAL_EPOCHS, IMG = [1], 1, 2, 128
    MODELS = {"SMOKE_UNet_ResNet18": lambda: smp.Unet("resnet18", encoder_weights=None, classes=1)}

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT_DIRS = {
    "train": ("training/training/Training_original", "training/training/Finalmasks"),
    "val": ("val/val/Original", "val/val/val_100masks"),
    "test": ("Testing/Testing/Original", "Testing/Testing/testing_100masks"),
}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"] = str(s)


# ------------------------------------------------------------------- DATA (as original)
def load_image(path):
    return np.asarray(Image.open(path).convert("RGB").resize((IMG, IMG), Image.BILINEAR), np.float32) / 255.0


def load_split(split, limit=None):
    img_dir, mask_dir = (os.path.join(DATA_ROOT, p) for p in SPLIT_DIRS[split])
    X, Y = [], []
    for f in sorted(os.listdir(mask_dir)):
        stem = f[: -len("_mask.png")]
        ip = os.path.join(img_dir, stem + ".jpg")
        if not os.path.exists(ip):
            continue
        X.append(load_image(ip))
        m = np.asarray(Image.open(os.path.join(mask_dir, f)).convert("L").resize((IMG, IMG), Image.NEAREST))
        Y.append((m > 127).astype(np.float32)[..., None])
        if limit and len(X) >= limit:
            break
    print(f"{split}: {len(X)} annotated images")
    return np.stack(X), np.stack(Y)


def augment(image, mask):
    """Same geometric + photometric augmentation as the original notebooks."""
    h, w = image.shape[:2]
    if np.random.rand() < 0.5:
        a = np.random.uniform(-15, 15)
        image = rotate(image, a, mode="reflect", preserve_range=True)
        mask = rotate(mask, a, mode="reflect", preserve_range=True)
    if np.random.rand() < 0.5:
        z = np.random.uniform(0.9, 1.1); nh, nw = int(h * z), int(w * z)
        iz = resize(image, (nh, nw), preserve_range=True, anti_aliasing=True)
        mz = resize(mask, (nh, nw), preserve_range=True, anti_aliasing=False, order=0)
        if z > 1.0:
            sh, sw = (nh - h) // 2, (nw - w) // 2
            image, mask = iz[sh:sh + h, sw:sw + w], mz[sh:sh + h, sw:sw + w]
        else:
            ph, pw = (h - nh) // 2, (w - nw) // 2
            pad = ((ph, h - nh - ph), (pw, w - nw - pw), (0, 0))
            image = np.pad(iz, pad, mode="reflect")
            mask = np.pad(mz, pad, mode="constant", constant_values=0)
    if np.random.rand() < 0.5:
        image, mask = np.fliplr(image), np.fliplr(mask)
    if np.random.rand() < 0.5:
        image, mask = np.flipud(image), np.flipud(mask)
    image = image + np.random.uniform(-0.05, 0.05)
    c, mu = np.random.uniform(0.95, 1.05), image.mean()
    image = (image - mu) * c + mu
    image = np.power(np.clip(image, 0, 1), np.random.uniform(0.95, 1.05))
    image = np.clip(image + np.random.normal(0, 0.005, image.shape), 0, 1)
    return image.astype(np.float32), mask.astype(np.float32)


class DS(torch.utils.data.Dataset):
    def __init__(self, X, Y, aug):
        self.X, self.Y, self.aug = X, Y, aug

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x, y = self.X[i], self.Y[i]
        if self.aug:
            x, y = augment(x, y)
        x = (x - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy((y > 0.5).astype(np.float32).transpose(2, 0, 1).copy())


def worker_init(wid):
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)


# ------------------------------------------------------------------ LOSSES (as original)
def dice_loss(p, y, eps=1e-6):
    return 1 - (2 * (p * y).sum() + eps) / (p.sum() + y.sum() + eps)


def tversky_loss(p, y, alpha, beta, eps=1e-6):
    tp = (p * y).sum(); fn = (y * (1 - p)).sum(); fp = ((1 - y) * p).sum()
    return 1 - (tp + eps) / (tp + alpha * fn + beta * fp + eps)


def focal_loss(p, y, alpha=0.75, gamma=2.0):
    p = p.clamp(1e-7, 1 - 1e-7)
    pt = torch.where(y == 1, p, 1 - p)
    at = torch.where(y == 1, torch.full_like(p, alpha), torch.full_like(p, 1 - alpha))
    return (-at * (1 - pt) ** gamma * torch.log(pt)).mean()


def loss_fn(logits, y, phase):
    p = torch.sigmoid(logits.float())
    if LOSS_MODE == "tfd":   # Eq. (15): equal weights, Tversky alpha=0.3, beta=0.7
        return tversky_loss(p, y, 0.3, 0.7) + focal_loss(p, y) + dice_loss(p, y)
    if phase == 1:           # Eq. (9)
        return 5 * dice_loss(p, y) + 3 * tversky_loss(p, y, 0.7, 0.3) + focal_loss(p, y)
    return 3 * dice_loss(p, y) + focal_loss(p, y)   # Eq. (10)


# ------------------------------------------------------------------------- TRAINING
@torch.no_grad()
def predict(model, X):
    model.eval(); out = []
    for i in range(0, len(X), BATCH_SIZE):
        x = torch.from_numpy(((X[i:i + BATCH_SIZE] - MEAN) / STD).transpose(0, 3, 1, 2).copy()).to(DEV)
        with torch.autocast(DEV, enabled=DEV == "cuda"):
            out.append(torch.sigmoid(model(x).float()).cpu().numpy()[:, 0])
    return np.concatenate(out)


def val_dice(model, Xv, Yv):
    p = predict(model, Xv); y = Yv[..., 0]
    return float(2 * (p * y).sum() / (p.sum() + y.sum() + 1e-6))   # soft Dice, as the Keras metric


def set_bn_eval(m):
    for mod in m.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.eval()


def run_phase(model, phase, loader, Xv, Yv, opt, start_ep, max_ep, best, sched_patience, min_lr):
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=sched_patience, min_lr=min_lr)
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
    best_d, best_state, wait = best[0], best[1], 0
    for ep in range(start_ep, max_ep):
        model.train()
        if phase == 2:
            set_bn_eval(model)   # BatchNorm frozen during fine-tuning, as in the original
        t0, tot = time.time(), 0.0
        for x, y in loader:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEV, enabled=DEV == "cuda"):
                logits = model(x)
            loss = loss_fn(logits, y, phase)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
        d = val_dice(model, Xv, Yv); sched.step(d)
        print(f"  phase{phase} ep{ep + 1:3d} loss={tot / len(loader):.4f} val_dice={d:.4f} "
              f"lr={opt.param_groups[0]['lr']:.1e} ({time.time() - t0:.0f}s)")
        if d > best_d:
            best_d, wait = d, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                print("  early stop"); break
    return ep + 1, (best_d, best_state)


def train_one(name, seed, data):
    Xtr, Ytr, Xv, Yv = data
    set_seed(seed)
    model = MODELS[name]().to(DEV)
    g = torch.Generator(); g.manual_seed(seed)
    loader = torch.utils.data.DataLoader(DS(Xtr, Ytr, True), batch_size=BATCH_SIZE, shuffle=True,
                                         num_workers=2, worker_init_fn=worker_init, generator=g, drop_last=False)
    # Phase 1: encoder frozen, Adam 1e-3
    for p_ in model.encoder.parameters():
        p_.requires_grad = False
    opt = torch.optim.Adam([p_ for p_ in model.parameters() if p_.requires_grad], lr=1e-3)
    ep, best = run_phase(model, 1, loader, Xv, Yv, opt, 0, PHASE1_EPOCHS, (-1, None), 5, 1e-6)
    model.load_state_dict(best[1])
    # Phase 2: encoder unfrozen (BN frozen), AdamW 5e-5, wd 1e-5
    for p_ in model.encoder.parameters():
        p_.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    _, best = run_phase(model, 2, loader, Xv, Yv, opt, ep, TOTAL_EPOCHS, best, 8, 1e-7)
    model.load_state_dict(best[1])
    return model, best[0]


def save_probs(model, out_dir, img_dir, keep):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")) and keep(f))
    if SMOKE:
        files = files[:3]
    for i in range(0, len(files), 16):
        chunk = files[i:i + 16]
        probs = predict(model, np.stack([load_image(os.path.join(img_dir, f)) for f in chunk]))
        for f, p in zip(chunk, probs):
            Image.fromarray(np.round(p * 255).astype(np.uint8)).save(os.path.join(out_dir, os.path.splitext(f)[0] + ".png"))


def export_run(model, run_dir, info):
    val_dir = os.path.join(DATA_ROOT, SPLIT_DIRS["val"][0])
    test_dir = os.path.join(DATA_ROOT, SPLIT_DIRS["test"][0])
    save_probs(model, os.path.join(run_dir, "val"), val_dir, lambda f: "Clean" not in f)
    save_probs(model, os.path.join(run_dir, "test"), test_dir, lambda f: "Clean" not in f)
    save_probs(model, os.path.join(run_dir, "clean"), test_dir, lambda f: "Clean" in f)
    with open(os.path.join(run_dir, "run_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)   # written last: marks the run as complete


# %%
def import_finished_runs():
    """Kaggle resume: copy finished runs from earlier versions' outputs (added via Add Input,
    read-only under /kaggle/input) into PRED_ROOT so they are skipped in this session."""
    import glob
    import shutil
    pats = ["/kaggle/input/*/preds", "/kaggle/input/*/*/preds", "/kaggle/input/*/*/*/preds"]
    for src in sorted({p for pat in pats for p in glob.glob(pat)}):
        for info in glob.glob(os.path.join(src, "*", "seed*", "run_info.json")):
            run_src = os.path.dirname(info)
            run_dst = os.path.join(PRED_ROOT, os.path.relpath(run_src, src))
            if not os.path.exists(os.path.join(run_dst, "run_info.json")):
                shutil.copytree(run_src, run_dst, dirs_exist_ok=True)
                print("imported finished run:", os.path.relpath(run_src, src))


if __name__ == "__main__":
    import_finished_runs()
    DATA_ROOT = find_data_root(DATA_ROOT)
    print("DATA_ROOT:", DATA_ROOT, "| models:", list(MODELS), "| seeds:", SEEDS)
    lim = 8 if SMOKE else None
    Xtr, Ytr = load_split("train", lim)
    Xv, Yv = load_split("val", lim)
    print("device:", DEV)
    for name in MODELS:
        for seed in SEEDS:
            run_dir = os.path.join(PRED_ROOT, name, f"seed{seed}")
            if os.path.exists(os.path.join(run_dir, "run_info.json")):
                print(f"skip {name} seed{seed} (done)"); continue
            print(f"\n=== {name} | seed {seed} | loss {LOSS_MODE} ===")
            try:
                t0 = time.time()
                model, best = train_one(name, seed, (Xtr, Ytr, Xv, Yv))
                n_params = sum(p.numel() for p in model.parameters())
                export_run(model, run_dir, dict(model=name, seed=seed, loss=LOSS_MODE, best_val_soft_dice=best,
                                                params_M=n_params / 1e6, train_minutes=(time.time() - t0) / 60))
                print(f"done {name} seed{seed}: best val dice {best:.4f}, {n_params / 1e6:.1f}M params")
            except Exception as e:  # keep going with the remaining models
                print(f"FAILED {name} seed{seed}: {type(e).__name__}: {e}")
            finally:
                model = None; gc.collect()
                if DEV == "cuda":
                    torch.cuda.empty_cache()
    # Then score everything (proposed model predictions exported into PRED_ROOT/Proposed/seed*/):
    # !python evaluate_quantification.py --data-root $DATA_ROOT --pred-root preds --out results_applied --reference Proposed
