"""
utils/satellite_generator.py  (v2 — realistic distributions)
--------------------------------------------------------------
Generates 20,000 synthetic 64×64×3 satellite image patches.

Why v1 hit 100 %
----------------
v1 hard-clipped SAR pixels:  flood ≥ 0.70, no-flood ≤ 0.40 — a 0.30 gap
with ZERO overlap.  Any depth-1 stump on image-mean SAR achieved 100 %.

v2 fix — three mechanisms (same strategy as data_generator.py v2)
------------------------------------------------------------------
1. Per-image center drawn from OVERLAPPING uniform ranges
   Flood    SAR center ~ U(0.45, 0.85)  mean=0.65
   No-flood SAR center ~ U(0.15, 0.60)  mean=0.375
   Overlap window [0.45, 0.60] — ~19 % of flood + ~17 % of no-flood
   images land in this ambiguous zone regardless of label.

2. Wider intra-image noise  (σ raised from 0.02→0.06 on SAR/NDVI)
   Each pixel deviates more from its image-mean, further blurring
   any single-pixel threshold that a model might exploit.

3. Label noise  (LABEL_NOISE_RATE = 0.05 → 5 %)
   500 flood images re-labelled 0, 500 no-flood images re-labelled 1.
   Features stay unchanged → hard examples, Bayes error floor ≈ 5 %.
   Class balance preserved (10 000 each after noise).

Expected achievable accuracy ceiling ≈ 95 %; realistic trained CNN ≈ 88–94 %.

Outputs
-------
  data/processed/satellite_images.npy   (20000, 64, 64, 3)  float32
  data/processed/image_labels.npy       (20000,)             int8
  data/satellite_images/sample_*.png
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

SEED             = 42
RNG              = np.random.default_rng(SEED)

N_TOTAL          = 20_000
N_FLOOD          = 10_000
N_NO_FLOOD       = 10_000
LABEL_NOISE_RATE = 0.05
IMG_SIZE         = 64
N_SAMPLES        = 5


# ════════════════════════════════════════════════════════════════════════
#  Spatial texture helper
# ════════════════════════════════════════════════════════════════════════

def _perlin_like(size: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    coarse = rng.random((max(4, size // 8), max(4, size // 8))).astype(np.float32)
    smooth = np.asarray(cv2.resize(coarse, (size, size), interpolation=cv2.INTER_LINEAR), dtype=np.float32)
    noise  = rng.normal(0, scale, (size, size)).astype(np.float32)
    return np.clip(smooth + noise, 0.0, 1.0)


def _gaussian_blob(canvas, cx, cy, radius, intensity, rng):
    H, W = canvas.shape
    mask = np.zeros((H, W), dtype=np.float32)
    axes  = (max(1, radius + rng.integers(-radius//3, radius//3+1)),
             max(1, radius + rng.integers(-radius//3, radius//3+1)))
    angle = int(rng.integers(0, 180))
    cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, intensity, -1)
    ksize  = max(3, (radius // 2) * 2 + 1)
    blurred = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=radius * 0.5)
    return np.clip(canvas + blurred, 0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════
#  Per-image generators  — overlapping distributions
# ════════════════════════════════════════════════════════════════════════

def _make_flood_image(rng: np.random.Generator) -> np.ndarray:
    """
    Flood patch.

    SAR center  ~ U(0.45, 0.85)  → ~19% of images land in the
                                    ambiguous [0.45, 0.60] zone
    NDVI center ~ U(0.02, 0.38)  → some images have NDVI up to 0.55
    DEM         : low-flat terrain, occasionally mid-elevation
    """
    S = IMG_SIZE

    # Ch 0  NDVI — suppressed, but center varies into borderline territory
    ndvi_center = float(rng.uniform(0.02, 0.38))
    ndvi_base   = _perlin_like(S, 0.06, rng) * 0.22 + ndvi_center
    ndvi_base   = np.clip(ndvi_base + rng.normal(0, 0.04, (S, S)), 0.0, 0.65)

    # Ch 1  SAR — high water return + blobs, center can be < 0.70
    sar_center = float(rng.uniform(0.45, 0.85))
    sar_base   = _perlin_like(S, 0.06, rng) * 0.22 + sar_center
    n_blobs    = int(rng.integers(1, 5))
    for _ in range(n_blobs):
        cx, cy  = int(rng.integers(8, S-8)), int(rng.integers(8, S-8))
        radius  = int(rng.integers(3, 14))
        sar_base = _gaussian_blob(sar_base, cx, cy, radius,
                                   float(rng.uniform(0.08, 0.22)), rng)
    sar_base = np.clip(sar_base + rng.normal(0, 0.05, (S, S)), 0.0, 1.0)

    # Ch 2  DEM — low, noisy
    dem_center = float(rng.uniform(0.05, 0.35))
    dem_base   = _perlin_like(S, 0.04, rng) * 0.22 + dem_center
    dem_base   = np.clip(dem_base + rng.normal(0, 0.03, (S, S)), 0.0, 0.65)

    return np.stack([ndvi_base, sar_base, dem_base], axis=-1).astype(np.float32)


def _make_noflood_image(rng: np.random.Generator) -> np.ndarray:
    """
    No-flood patch.

    SAR center  ~ U(0.15, 0.60)  → ~17% of images land in the
                                    ambiguous [0.45, 0.60] zone
    NDVI center ~ U(0.38, 0.75)  → some images have NDVI down to 0.30
    DEM         : varied elevation
    """
    S = IMG_SIZE

    # Ch 0  NDVI — healthy vegetation, center can dip into borderline range
    ndvi_center = float(rng.uniform(0.38, 0.75))
    ndvi_base   = _perlin_like(S, 0.06, rng) * 0.30 + ndvi_center
    ndvi_base   = np.clip(ndvi_base + rng.normal(0, 0.05, (S, S)), 0.10, 1.0)

    # Ch 1  SAR — dry return, center can rise into borderline range (> 0.40)
    sar_center = float(rng.uniform(0.15, 0.60))
    sar_base   = _perlin_like(S, 0.07, rng) * 0.25 + sar_center
    sar_base   = np.clip(sar_base + rng.normal(0, 0.05, (S, S)), 0.0, 1.0)

    # Ch 2  DEM — varied
    dem_center = float(rng.uniform(0.20, 0.80))
    dem_base   = _perlin_like(S, 0.05, rng) * 0.40 + dem_center
    dem_base   = np.clip(dem_base + rng.normal(0, 0.04, (S, S)), 0.0, 1.0)

    return np.stack([ndvi_base, sar_base, dem_base], axis=-1).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════
#  Batch generator
# ════════════════════════════════════════════════════════════════════════

def _generate_batch(generator_fn, n, rng, label, desc):
    images = np.empty((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    bar_w  = 38
    for i in range(n):
        images[i] = generator_fn(rng)
        if (i + 1) % 500 == 0 or (i + 1) == n:
            done = int((i+1)/n*bar_w)
            print(f"\r  {desc}  [{'█'*done}{'░'*(bar_w-done)}]"
                  f"  {i+1:,}/{n:,}  {(i+1)/n*100:.1f}%", end="", flush=True)
    print()
    return images, np.full(n, label, dtype=np.int8)


# ════════════════════════════════════════════════════════════════════════
#  Label noise  (identical logic to data_generator.py)
# ════════════════════════════════════════════════════════════════════════

def _apply_label_noise(flood_labels, noflood_labels, noise_rate, rng):
    n_flip     = int(len(flood_labels) * noise_rate)
    fi         = rng.choice(len(flood_labels),   n_flip, replace=False)
    ni         = rng.choice(len(noflood_labels), n_flip, replace=False)
    flood_labels[fi]   = 0
    noflood_labels[ni] = 1
    return flood_labels, noflood_labels


# ════════════════════════════════════════════════════════════════════════
#  PNG sample saver
# ════════════════════════════════════════════════════════════════════════

def _save_sample_pngs(images, labels, out_dir, n_samples=5):
    out_dir  = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ch_names = ["Ch 0 – NDVI", "Ch 1 – SAR", "Ch 2 – DEM"]
    ch_cmaps = ["RdYlGn", "Blues", "terrain"]

    for cls, cls_name in ((1, "flood"), (0, "noflood")):
        for rank, idx in enumerate(np.where(labels == cls)[0][:n_samples]):
            img = images[idx]
            fig = plt.figure(figsize=(13, 4.2), facecolor="#1a1a2e")
            fig.suptitle(f"{'FLOOD' if cls==1 else 'NO-FLOOD'} sample {rank}",
                         color="white", fontsize=13, fontweight="bold", y=1.01)
            gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35,
                                    left=0.04, right=0.96)
            for ch in range(3):
                ax = fig.add_subplot(gs[0, ch])
                im = ax.imshow(img[..., ch], cmap=ch_cmaps[ch],
                               vmin=0, vmax=1, interpolation="nearest")
                ax.set_title(ch_names[ch], color="white", fontsize=9, pad=4)
                ax.axis("off")
                cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.ax.yaxis.set_tick_params(color="white", labelsize=7)
                plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
                cb.outline.set_edgecolor("white")
            ax_s = fig.add_subplot(gs[0, 3])
            ax_s.set_facecolor("#0d1117"); ax_s.axis("off")
            lines = ["PIXEL STATS\n"]
            for ch, n in enumerate(["NDVI","SAR ","DEM "]):
                d = img[..., ch].ravel()
                lines.append(f"{n}\n  mean {d.mean():.3f}\n  std  {d.std():.3f}\n"
                              f"  min  {d.min():.3f}\n  max  {d.max():.3f}\n")
            ax_s.text(0.05, 0.95, "\n".join(lines), transform=ax_s.transAxes,
                      color="white", fontsize=8, verticalalignment="top",
                      fontfamily="monospace",
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e",
                                edgecolor="#444"))
            fig.savefig(str(out_dir / f"sample_{cls_name}_{rank}.png"),
                        dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
    print(f"  Saved {n_samples*2} sample PNGs → {out_dir}")


# ════════════════════════════════════════════════════════════════════════
#  Verification report
# ════════════════════════════════════════════════════════════════════════

def _print_report(images, labels, proc_dir, img_dir):
    flood = images[labels == 1]; nofl = images[labels == 0]

    # Image-level SAR means
    flood_sar_means = flood[:, :, :, 1].mean(axis=(1, 2))
    nofl_sar_means  = nofl[:, :, :, 1].mean(axis=(1, 2))

    print("\n" + "=" * 64)
    print("  SATELLITE DATASET VERIFICATION REPORT  (v2 — realistic)")
    print("=" * 64)

    print(f"\n  CLASS DISTRIBUTION  (after {LABEL_NOISE_RATE:.0%} label noise)")
    for lbl, cnt in sorted(zip(*np.unique(labels, return_counts=True))):
        name = "Flood (1)" if lbl == 1 else "No-Flood (0)"
        bar  = "█" * (cnt // 400)
        print(f"  {name:<13} {cnt:>7,}  50.00%  {bar}")
    print(f"  {'TOTAL':<13} {N_TOTAL:>7,}")

    print(f"\n  IMAGE-MEAN SAR STATISTICS  (key diagnostic)")
    print(f"  {'Class':<12} {'mean':>7} {'std':>7} {'min':>7} {'max':>7}"
          f"  {'p10':>7} {'p90':>7}")
    print("  " + "─" * 60)
    for name, arr in [("Flood", flood_sar_means), ("No-Flood", nofl_sar_means)]:
        print(f"  {name:<12} {arr.mean():>7.3f} {arr.std():>7.3f}"
              f" {arr.min():>7.3f} {arr.max():>7.3f}"
              f"  {np.percentile(arr,10):>7.3f} {np.percentile(arr,90):>7.3f}")

    overlap = ((flood_sar_means > nofl_sar_means.min()) &
               (flood_sar_means < nofl_sar_means.max())).mean()
    print(f"\n  SAR overlap zone: {overlap*100:.1f}% of flood images"
          f" overlap with no-flood range")

    # Depth-1 stump accuracy estimate
    from sklearn.tree import DecisionTreeClassifier
    feat = images.mean(axis=(1, 2))                          # (N, 3)
    dt   = DecisionTreeClassifier(max_depth=1, random_state=42)
    dt.fit(feat, labels)
    stump_acc = dt.score(feat, labels)
    print(f"  Depth-1 stump accuracy on image means: {stump_acc*100:.1f}%")
    print(f"  (CNN target: ~88–94%  |  Bayes ceiling: ~{(1-LABEL_NOISE_RATE)*100:.0f}%)")

    print(f"\n  LABEL NOISE INJECTED")
    print(f"  Flood→0:    {int(N_FLOOD*LABEL_NOISE_RATE):>5,} images")
    print(f"  No-flood→1: {int(N_NO_FLOOD*LABEL_NOISE_RATE):>5,} images")

    img_npy = Path(proc_dir) / "satellite_images.npy"
    lbl_npy = Path(proc_dir) / "image_labels.npy"
    print(f"\n  SAVED FILES")
    print(f"  {img_npy}  ({img_npy.stat().st_size/1e6:.1f} MB)")
    print(f"  {lbl_npy}  ({lbl_npy.stat().st_size/1e3:.1f} KB)")
    print("=" * 64 + "\n")


# ════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════

def generate_satellite_dataset(
    processed_dir: str = "data/processed",
    images_dir:    str = "data/satellite_images",
) -> tuple[np.ndarray, np.ndarray]:

    proc_dir = Path(processed_dir); proc_dir.mkdir(parents=True, exist_ok=True)
    img_dir  = Path(images_dir);   img_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  Satellite Image Generator  v2  (realistic distributions)")
    print(f"  Seed={SEED}  |  N={N_TOTAL:,}  |  "
          f"LabelNoise={LABEL_NOISE_RATE:.0%}  |  Size={IMG_SIZE}×{IMG_SIZE}×3")
    print("=" * 64 + "\n")

    print("[1/4]  Generating flood images …")
    flood_imgs, flood_lbs = _generate_batch(
        _make_flood_image, N_FLOOD, RNG, 1, "Flood   ")

    print("[2/4]  Generating no-flood images …")
    nofl_imgs, nofl_lbs = _generate_batch(
        _make_noflood_image, N_NO_FLOOD, RNG, 0, "No-Flood")

    print(f"\n[3/4]  Applying {LABEL_NOISE_RATE:.0%} label noise …")
    flood_lbs, nofl_lbs = _apply_label_noise(
        flood_lbs, nofl_lbs, LABEL_NOISE_RATE, RNG)

    print("[4/4]  Concatenating, shuffling, saving …")
    images = np.concatenate([flood_imgs, nofl_imgs], axis=0)
    labels = np.concatenate([flood_lbs,  nofl_lbs],  axis=0)
    shuf   = RNG.permutation(N_TOTAL)
    images, labels = images[shuf], labels[shuf]

    np.save(proc_dir / "satellite_images.npy", images)
    np.save(proc_dir / "image_labels.npy",     labels)
    _save_sample_pngs(images, labels, img_dir, N_SAMPLES)
    _print_report(images, labels, proc_dir, img_dir)
    return images, labels


if __name__ == "__main__":
    generate_satellite_dataset()
