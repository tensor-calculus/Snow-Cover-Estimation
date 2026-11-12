import os
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

FIG_DIR = r"C:\Users\rnine\LIS\report\figures"
FIG_DIR_EXTRACTED = r"C:\Users\rnine\LIS\archives\Phase_I_extracted\figures"
ARTIFACT_DIR = r"C:\Users\rnine\.gemini\antigravity\brain\6d8b186a-6c3e-4116-b7ca-7e2c8ad7932c"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(FIG_DIR_EXTRACTED, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 11,
    'axes.titlesize': 12.5,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11.5,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'figure.dpi': 300
})


def save_all(fig, basename):
    """Save both PNG and PDF to figures/, Phase_I_extracted/figures/, and artifact dir."""
    for d in [FIG_DIR, FIG_DIR_EXTRACTED, ARTIFACT_DIR]:
        if os.path.isdir(d):
            fig.savefig(os.path.join(d, f"{basename}.png"), dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(d, f"{basename}.pdf"), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {basename}.png and .pdf")


def plot_snowline_domain_shift(data_dir):
    """Plot 1: Histogram of snow pixel elevations across Train, Val, and Test splits."""
    with rasterio.open(os.path.join(data_dir, "DEM_aligned.tif")) as s_dem, \
         rasterio.open(os.path.join(data_dir, "patch7", "training_labels.tif")) as s_tr, \
         rasterio.open(os.path.join(data_dir, "patch12", "training_labels.tif")) as s_val:
        dem_main = s_dem.read(1, out_shape=(s_dem.height // 4, s_dem.width // 4))
        lbl_tr = s_tr.read(1, out_shape=(s_tr.height // 4, s_tr.width // 4))
        lbl_val = s_val.read(1, out_shape=(s_val.height // 4, s_val.width // 4))

    test_dir = os.path.join(data_dir, "patch_test", "winter_img")
    with rasterio.open(os.path.join(test_dir, "DEM_aligned.tif")) as s_tdem, \
         rasterio.open(os.path.join(test_dir, "training_labels.tif")) as s_tlbl:
        dem_test = s_tdem.read(1, out_shape=(s_tdem.height // 4, s_tdem.width // 4))
        lbl_test = s_tlbl.read(1, out_shape=(s_tlbl.height // 4, s_tlbl.width // 4))

    elev_train_snow = dem_main[(lbl_tr == 1) & (dem_main > 0)]
    elev_val_snow = dem_main[(lbl_val == 1) & (dem_main > 0)]
    elev_test_snow = dem_test[(lbl_test == 1) & (dem_test > 0)]

    fig, ax = plt.subplots(figsize=(6.8, 3.6), constrained_layout=True)

    bins = np.linspace(2500, 7000, 55)
    ax.hist(elev_train_snow, bins=bins, density=True, alpha=0.45, color='#1E88E5',
            label='Train Region (patch7–11, Median: 5,169 m)')
    ax.hist(elev_val_snow, bins=bins, density=True, histtype='step', linewidth=2.2, color='#0D47A1',
            label='Val Region (patch12, Median: 5,180 m)')
    ax.hist(elev_test_snow, bins=bins, density=True, alpha=0.50, color='#E53935',
            label='Test Region (patch_test, Median: 4,298 m)')

    ax.axvline(5169, color='#1565C0', linestyle='--', linewidth=2.0)
    ax.axvline(4298, color='#C62828', linestyle='--', linewidth=2.0)

    ax.annotate('', xy=(4298, 0.00070), xytext=(5169, 0.00070),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2.0))
    ax.text((4298 + 5169) / 2, 0.00075, '871 m Snowline Shift',
            fontsize=10.5, fontweight='bold', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFF9C4', edgecolor='#FBC02D'))

    ax.set_title("Snow Elevation Distribution: Train/Val vs. Unseen Test Region")
    ax.set_xlabel("Elevation (meters above sea level)")
    ax.set_ylabel("Normalized Snow Pixel Density")
    ax.set_ylim(0, 0.00115)
    ax.legend(loc='upper right', fontsize=9.8)
    ax.grid(True, linestyle=':', alpha=0.6)

    save_all(fig, "domain_shift_plot")


def plot_elevation_performance():
    """Plot 2: Grouped bar chart of Model 1, 2, and 3 performance across elevation zones and splits."""
    categories = [
        "<3000 m\n(Valleys Acc)",
        "3000–4000 m\n(Lower F1)",
        "4000–5000 m\n(Mid F1)",
        ">5000 m\n(High F1)",
        "Validation\n(Global F1)",
        "Test\n(Global F1)"
    ]
    m1_scores = [96.42, 25.35, 50.97, 73.08, 56.84, 43.34]
    m2_scores = [99.93, 24.73, 51.32, 66.79, 67.10, 46.46]
    m3_scores = [100.00, 25.87, 55.35, 70.29, 69.27, 50.42]

    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)

    x = np.arange(len(categories))
    width = 0.26

    b1 = ax.bar(x - width, m1_scores, width, label='Model 1: 2-ch (Winter SAR)', color='#90CAF9', edgecolor='#1565C0')
    b2 = ax.bar(x, m2_scores, width, label='Model 2: 4-ch (+ Summer Ref)', color='#FFB74D', edgecolor='#E65100')
    b3 = ax.bar(x + width, m3_scores, width, label='Model 3: 6-ch (+ DEM & Slope)', color='#66BB6A', edgecolor='#1B5E20')

    for b_idx, bars in enumerate([b1, b2, b3]):
        for i, bar in enumerate(bars):
            h = bar.get_height()
            y_off = 7 if (b_idx == 1 and i in [0, 1, 2]) else 2
            ax.annotate(f'{h:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, y_off), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.8, fontweight='bold')

    ax.set_title("Model Performance Across Elevation Zones and Dataset Splits")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10.0, fontweight='bold')
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 118)
    ax.legend(loc='upper right', fontsize=9.8)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)

    save_all(fig, "elevation_performance_plot")


def plot_scene_cloud_and_snow_stats(data_dir):
    """Plot 3: Cloud obscuration vs. valid snow percentage across all winter scenes."""
    scenes = [
        ("patch7\n(09 Dec)", os.path.join(data_dir, "patch7", "training_labels.tif")),
        ("patch8\n(21 Dec)", os.path.join(data_dir, "patch8", "training_labels.tif")),
        ("patch9\n(02 Jan)", os.path.join(data_dir, "patch9", "training_labels.tif")),
        ("patch10\n(14 Jan)", os.path.join(data_dir, "patch10", "training_labels.tif")),
        ("patch11\n(26 Jan)", os.path.join(data_dir, "patch11", "training_labels.tif")),
        ("patch12\n(19 Feb Val)", os.path.join(data_dir, "patch12", "training_labels.tif")),
        ("patch_test\n(21 Jan Test)", os.path.join(data_dir, "patch_test", "winter_img", "training_labels.tif")),
    ]
    labels = []
    snow_pct = []
    cloud_pct = []

    for name, path in scenes:
        with rasterio.open(path) as src:
            a = src.read(1, out_shape=(1500, 1500))
        valid_swath = (a != 255)
        total_swath = max(1, np.sum(valid_swath))
        c_pct = np.sum(a == 2) / total_swath * 100.0
        s_pct = np.sum(a == 1) / max(1, np.sum((a == 0) | (a == 1))) * 100.0
        labels.append(name)
        snow_pct.append(s_pct)
        cloud_pct.append(c_pct)

    fig, ax = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.34

    b1 = ax.bar(x - width / 2, snow_pct, width, label='Valid Snow Cover (% of Clear-Sky Pixels)',
                color='#29B6F6', edgecolor='#0277BD')
    b2 = ax.bar(x + width / 2, cloud_pct, width, label='Optical Cloud Obscuration (% of Scene Masked)',
                color='#BDBDBD', edgecolor='#424242', hatch='//')

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.8, fontweight='bold')

    ax.set_title("Winter Scene Snow Cover vs. Optical Cloud Obscuration")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.8, fontweight='bold')
    ax.set_ylabel("Percentage of Scene (%)")
    ax.set_ylim(0, 48)
    ax.legend(loc='upper left', fontsize=9.8)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)

    save_all(fig, "scene_cloud_snow_plot")


def plot_sar_backscatter_distributions(data_dir):
    """Plot 4: Physical C-band SAR backscatter difference (Winter - Summer MLE Ref in dB) for Snow vs. No-Snow."""
    s1_path = os.path.join(data_dir, "patch12", "S1A_Terrain_Corrected.tif")
    mle_path = os.path.join(data_dir, "S1A_MLE_Reference.tif")
    lbl_path = os.path.join(data_dir, "patch12", "training_labels.tif")

    with rasterio.open(s1_path) as s_s1, \
         rasterio.open(mle_path) as s_mle, \
         rasterio.open(lbl_path) as s_lbl:
        out_s = (s_lbl.height // 8, s_lbl.width // 8)
        vh_w = s_s1.read(1, out_shape=out_s).astype(np.float32)
        vv_w = s_s1.read(2, out_shape=out_s).astype(np.float32)
        vh_r = s_mle.read(1, out_shape=out_s).astype(np.float32)
        vv_r = s_mle.read(2, out_shape=out_s).astype(np.float32)
        lbl = s_lbl.read(1, out_shape=out_s)

    valid = (lbl <= 1) & np.isfinite(vv_w) & np.isfinite(vv_r) & (vv_w > 1e-4) & (vv_r > 1e-4) & (vh_w > 1e-4) & (vh_r > 1e-4)
    delta_vv = 10.0 * np.log10(vv_w[valid]) - 10.0 * np.log10(vv_r[valid])
    delta_vh = 10.0 * np.log10(vh_w[valid]) - 10.0 * np.log10(vh_r[valid])
    y_val = lbl[valid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)
    bins = np.linspace(-10, 8, 70)

    ax1.hist(delta_vv[y_val == 0], bins=bins, density=True, alpha=0.45, color='#8D6E63',
             label='No-Snow Ground (0)')
    ax1.hist(delta_vv[y_val == 1], bins=bins, density=True, alpha=0.55, color='#1E88E5',
             label='Snow Cover (1)')
    ax1.axvline(-2.0, color='#C62828', linestyle='--', linewidth=1.6, label='Wet-Snow Threshold (-2 dB)')
    ax1.set_title(r"(a) Co-Pol Contrast: $\Delta\sigma^0_{\mathrm{VV}} = \sigma^0_{\mathrm{VV},t} - \sigma^0_{\mathrm{VV,ref}}$")
    ax1.set_xlabel(r"$\Delta\sigma^0_{\mathrm{VV}}$ (dB)")
    ax1.set_ylabel("Probability Density")
    ax1.legend(loc='upper right', fontsize=8.8)
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.hist(delta_vh[y_val == 0], bins=bins, density=True, alpha=0.45, color='#8D6E63',
             label='No-Snow Ground (0)')
    ax2.hist(delta_vh[y_val == 1], bins=bins, density=True, alpha=0.55, color='#00897B',
             label='Snow Cover (1)')
    ax2.axvline(-2.0, color='#C62828', linestyle='--', linewidth=1.6, label='Wet-Snow Threshold (-2 dB)')
    ax2.set_title(r"(b) Cross-Pol Contrast: $\Delta\sigma^0_{\mathrm{VH}} = \sigma^0_{\mathrm{VH},t} - \sigma^0_{\mathrm{VH,ref}}$")
    ax2.set_xlabel(r"$\Delta\sigma^0_{\mathrm{VH}}$ (dB)")
    ax2.set_ylabel("Probability Density")
    ax2.legend(loc='upper right', fontsize=8.8)
    ax2.grid(True, linestyle=':', alpha=0.6)

    save_all(fig, "sar_backscatter_distribution_plot")


def plot_precision_recall_comparison():
    """Plot 5: Precision, Recall, F1_Snow, and Macro F1 breakdown on Validation and Unseen Test."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.5), constrained_layout=True)

    metrics = ["Snow\nPrecision", "Snow\nRecall", "Snow\nF1-Score", "Macro\nF1-Score"]
    x = np.arange(len(metrics))
    w = 0.25

    # Validation (patch12)
    val_m1 = [54.73, 59.12, 56.84, 77.02]
    val_m2 = [72.21, 62.66, 67.10, 82.58]
    val_m3 = [66.39, 72.42, 69.27, 83.68]

    for idx, (scores, lbl, col, ecol) in enumerate([
        (val_m1, 'Model 1: 2-ch SAR', '#90CAF9', '#1565C0'),
        (val_m2, 'Model 2: 4-ch + MLE', '#FFB74D', '#E65100'),
        (val_m3, 'Model 3: 6-ch + DEM', '#66BB6A', '#1B5E20'),
    ]):
        bars = ax1.bar(x + (idx - 1) * w, scores, w, label=lbl, color=col, edgecolor=ecol)
        for b in bars:
            h = b.get_height()
            ax1.annotate(f'{h:.1f}', xy=(b.get_x() + b.get_width() / 2, h),
                         xytext=(0, 2), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8.2, fontweight='bold')

    ax1.set_title("(a) Validation Basin (patch12)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=9.8, fontweight='bold')
    ax1.set_ylabel("Metric Score (%)")
    ax1.set_ylim(0, 102)
    ax1.legend(loc='upper left', fontsize=8.5)
    ax1.grid(True, axis='y', linestyle=':', alpha=0.6)

    # Test (patch_test)
    tst_m1 = [49.63, 38.47, 43.34, 68.18]
    tst_m2 = [82.97, 32.26, 46.46, 70.79]
    tst_m3 = [79.35, 36.95, 50.42, 72.82]

    for idx, (scores, lbl, col, ecol) in enumerate([
        (tst_m1, 'Model 1: 2-ch SAR', '#90CAF9', '#1565C0'),
        (tst_m2, 'Model 2: 4-ch + MLE', '#FFB74D', '#E65100'),
        (tst_m3, 'Model 3: 6-ch + DEM', '#66BB6A', '#1B5E20'),
    ]):
        bars = ax2.bar(x + (idx - 1) * w, scores, w, label=lbl, color=col, edgecolor=ecol)
        for b in bars:
            h = b.get_height()
            ax2.annotate(f'{h:.1f}', xy=(b.get_x() + b.get_width() / 2, h),
                         xytext=(0, 2), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8.2, fontweight='bold')

    ax2.set_title("(b) Unseen Spatial Transfer (patch_test)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=9.8, fontweight='bold')
    ax2.set_ylabel("Metric Score (%)")
    ax2.set_ylim(0, 102)
    ax2.legend(loc='upper right', fontsize=8.5)
    ax2.grid(True, axis='y', linestyle=':', alpha=0.6)

    save_all(fig, "precision_recall_comparison_plot")


def plot_threshold_sensitivity():
    """Plot 6: Diagnostic probability threshold sweep on patch_test showing domain-shift impact."""
    taus = [0.10, 0.20, 0.30, 0.50, 0.65]
    f1_snow = [62.17, 60.93, 58.83, 52.91, 46.46]
    prec_snow = [60.52, 66.70, 71.08, 78.05, 82.97]
    rec_snow = [63.92, 56.07, 50.18, 40.02, 32.26]
    overall_f1 = [90.76, 91.03, 90.91, 90.19, 89.27]

    fig, ax = plt.subplots(figsize=(6.6, 3.4), constrained_layout=True)
    ax.plot(taus, f1_snow, 'o-', lw=2.2, color='#1E88E5', label=r'$\mathrm{F1}_{\mathrm{Snow}}$ (%)')
    ax.plot(taus, prec_snow, 's--', lw=1.9, color='#2E7D32', label='Snow Precision (%)')
    ax.plot(taus, rec_snow, '^--', lw=1.9, color='#E65100', label='Snow Recall (%)')
    ax.plot(taus, overall_f1, 'd:', lw=1.8, color='#6A1B9A', label='Overall F1 (Eq. 3) (%)')

    ax.axvline(0.65, color='#C62828', linestyle='--', lw=1.6)
    ax.annotate('Locked Val Threshold\n($\\tau^* = 0.65$)', xy=(0.65, 46.46), xytext=(0.45, 58.0),
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5),
                fontsize=9.2, fontweight='bold', color='#C62828')

    ax.set_title("Probability Threshold Sensitivity Under 871 m Snowline Shift (patch_test)")
    ax.set_xlabel(r"Classification Probability Threshold ($\tau$)")
    ax.set_ylabel("Metric Score (%)")
    ax.set_ylim(18, 98)
    ax.set_xlim(0.06, 0.70)
    ax.legend(loc='lower left', fontsize=9.0, ncol=2)
    ax.grid(True, linestyle=':', alpha=0.6)

    save_all(fig, "threshold_sensitivity_plot")


def plot_s2_optical_lis_demo(data_dir):
    """Satellite Panel 1: Real Sentinel-2 L2A Optical RGB, SWIR False-Color, NDSI, and LIS Weak Labels."""
    granules = glob.glob(os.path.join(
        data_dir, "patch_test", "patch5", "sentinel2_extracted",
        "*T46RET*", "GRANULE", "*", "IMG_DATA", "R20m"
    ))
    if not granules:
        print("Skipping S2 optical demo (granule not found)")
        return
    r20 = granules[0]
    b03_path = glob.glob(os.path.join(r20, "*_B03_20m.jp2"))[0]
    b04_path = glob.glob(os.path.join(r20, "*_B04_20m.jp2"))[0]
    b11_path = glob.glob(os.path.join(r20, "*_B11_20m.jp2"))[0]
    tci_path = glob.glob(os.path.join(r20, "*_TCI_20m.jp2"))[0]
    lis_path = glob.glob(os.path.join(
        data_dir, "patch_test", "patch5", "lis_output", "*T46RET*", "LIS_S2-SNOW-FSC-TOC*.tif"
    ))[0]

    # Find a 1200x1200 window (24 km x 24 km at 20m) that is 100% inside the sensor swath (no nodata 255)
    # and has balanced Snow, No-Snow, and Clouds
    with rasterio.open(lis_path) as s_lis:
        full_lis = s_lis.read(1)
    best_score = -1
    best_rc = (1800, 1800)
    win_sz = 1200
    for r in range(200, full_lis.shape[0] - win_sz - 200, 300):
        for c in range(800, full_lis.shape[1] - win_sz - 200, 300):
            sub = full_lis[r:r+win_sz, c:c+win_sz]
            if np.mean(sub == 255) > 0.001:
                continue
            f_snow = np.mean((sub >= 1) & (sub <= 100))
            f_nosnow = np.mean(sub == 0)
            f_cloud = np.mean(sub == 205)
            if f_snow > 0.15 and f_nosnow > 0.20 and f_cloud > 0.15:
                score = min(f_snow, f_nosnow, f_cloud)
                if score > best_score:
                    best_score = score
                    best_rc = (r, c)

    r0, c0 = best_rc
    win = Window(c0, r0, win_sz, win_sz)
    with rasterio.open(tci_path) as s_tci, \
         rasterio.open(b03_path) as s_b03, \
         rasterio.open(b04_path) as s_b04, \
         rasterio.open(b11_path) as s_b11, \
         rasterio.open(lis_path) as s_lis:
        tci = np.moveaxis(s_tci.read([1, 2, 3], window=win), 0, -1)
        b03 = s_b03.read(1, window=win).astype(np.float32) / 10000.0
        b04 = s_b04.read(1, window=win).astype(np.float32) / 10000.0
        b11 = s_b11.read(1, window=win).astype(np.float32) / 10000.0
        lis = s_lis.read(1, window=win)

    # Compute SWIR composite (R: B11, G: B04, B: B03) and NDSI
    swir_rgb = np.stack([
        np.clip(b11 / 0.45, 0, 1),
        np.clip(b04 / 0.65, 0, 1),
        np.clip(b03 / 0.65, 0, 1)
    ], axis=-1)
    ndsi = (b03 - b11) / (b03 + b11 + 1e-6)

    # Discrete LIS class map: 0 = No-Snow, 1 = Snow (1..100), 2 = Cloud (205)
    lis_cls = np.zeros_like(lis, dtype=np.uint8)
    lis_cls[(lis >= 1) & (lis <= 100)] = 1
    lis_cls[lis == 205] = 2

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.6), constrained_layout=True)

    axes[0, 0].imshow(tci)
    axes[0, 0].set_title("(a) Sentinel-2 True Color\n(Snow & Clouds Both White)", fontsize=11.0, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(swir_rgb)
    axes[0, 1].set_title("(b) SWIR Composite (B11/B4/B3)\n(Snow = Cyan, Clouds = White)", fontsize=11.0, fontweight='bold')
    axes[0, 1].axis('off')

    im_ndsi = axes[1, 0].imshow(ndsi, cmap='RdYlBu', vmin=-0.5, vmax=0.8)
    axes[1, 0].set_title(r"(c) Spectral NDSI Map" + "\n" + r"$(\mathrm{B03}-\mathrm{B11})/(\mathrm{B03}+\mathrm{B11})$", fontsize=11.0, fontweight='bold')
    axes[1, 0].axis('off')
    cbar = fig.colorbar(im_ndsi, ax=axes[1, 0], fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=9.5)

    cmap_lis = ListedColormap(['#4E342E', '#29B6F6', '#E0E0E0'])
    axes[1, 1].imshow(lis_cls, cmap=cmap_lis, vmin=0, vmax=2, interpolation='nearest')
    axes[1, 1].set_title("(d) Let-It-Snow (LIS) Mask\n(Two-Pass NDSI + Cloud Mask)", fontsize=11.0, fontweight='bold')
    axes[1, 1].axis('off')
    legend_patches = [
        Patch(facecolor='#29B6F6', edgecolor='k', label='Snow (1)'),
        Patch(facecolor='#4E342E', edgecolor='k', label='No-Snow (0)'),
        Patch(facecolor='#E0E0E0', edgecolor='k', label='Cloud (255)'),
    ]
    axes[1, 1].legend(handles=legend_patches, loc='lower right', fontsize=9.5, framealpha=0.94)

    save_all(fig, "sentinel2_lis_pipeline_sample")


def plot_multitemporal_winter_series(data_dir):
    """Satellite Panel 2: Multi-temporal winter progression across patch7..patch12 (SAR VV vs. S2 LIS)."""
    patches = [
        ("patch7 (09 Dec)", "patch7"),
        ("patch8 (21 Dec)", "patch8"),
        ("patch9 (02 Jan)", "patch9"),
        ("patch10 (14 Jan)", "patch10"),
        ("patch11 (26 Jan)", "patch11"),
        ("patch12 (19 Feb)", "patch12"),
    ]
    # Automatically locate a 2048x2048 window with rich alpine snow (>30% in patch12) and zero swath nodata
    p12_lbl_path = os.path.join(data_dir, "patch12", "training_labels.tif")
    p9_lbl_path = os.path.join(data_dir, "patch9", "training_labels.tif")
    with rasterio.open(p12_lbl_path) as s12, rasterio.open(p9_lbl_path) as s9:
        H, W = s12.height, s12.width
        scale = 16
        a12 = s12.read(1, out_shape=(H // scale, W // scale))
        a9 = s9.read(1, out_shape=(H // scale, W // scale))

    ws = 2048 // scale
    best_score = -1
    best_rc = (H // 2, W // 2)
    for r in range(0, a12.shape[0] - ws, 16):
        for c in range(0, a12.shape[1] - ws, 16):
            sub12 = a12[r:r+ws, c:c+ws]
            sub9 = a9[r:r+ws, c:c+ws]
            if np.mean(sub12 == 255) > 0.01 or np.mean(sub9 == 255) > 0.01:
                continue
            snow12 = np.mean(sub12 == 1)
            nosnow12 = np.mean(sub12 == 0)
            cloud9 = np.mean(sub9 == 2)
            if snow12 > 0.30 and nosnow12 > 0.25:
                score = min(snow12, nosnow12) + 0.3 * cloud9
                if score > best_score:
                    best_score = score
                    best_rc = (r * scale, c * scale)

    win = Window(best_rc[1], best_rc[0], 2048, 2048)
    out_sz = (512, 512)

    fig, axes = plt.subplots(2, 6, figsize=(12.0, 4.3), constrained_layout=True)
    cmap_lis = ListedColormap(['#4E342E', '#29B6F6', '#D6D6D6'])

    for col_idx, (title, p_folder) in enumerate(patches):
        s1_p = os.path.join(data_dir, p_folder, "S1A_Terrain_Corrected.tif")
        lbl_p = os.path.join(data_dir, p_folder, "training_labels.tif")

        with rasterio.open(s1_p) as s_s1, rasterio.open(lbl_p) as s_lbl:
            vv = s_s1.read(2, window=win, out_shape=out_sz).astype(np.float32)
            lbl = s_lbl.read(1, window=win, out_shape=out_sz)

        vv_db = 10.0 * np.log10(np.clip(np.nan_to_num(vv, nan=1e-4), 1e-4, None))
        lbl_vis = np.zeros_like(lbl, dtype=np.uint8)
        lbl_vis[lbl == 1] = 1
        lbl_vis[(lbl == 2) | (lbl == 255)] = 2

        ax_top = axes[0, col_idx]
        ax_top.imshow(vv_db, cmap='gray', vmin=-22, vmax=0)
        ax_top.set_title(title, fontsize=9.8, fontweight='bold')
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        if col_idx == 0:
            ax_top.set_ylabel("Sentinel-1 SAR\n" + r"$\sigma^0_{\mathrm{VV}}$ (dB)", fontsize=10.0, fontweight='bold')

        ax_bot = axes[1, col_idx]
        ax_bot.imshow(lbl_vis, cmap=cmap_lis, vmin=0, vmax=2, interpolation='nearest')
        ax_bot.set_xticks([])
        ax_bot.set_yticks([])
        if col_idx == 0:
            ax_bot.set_ylabel("Sentinel-2 LIS\nWeak Labels", fontsize=10.0, fontweight='bold')

    legend_patches = [
        Patch(facecolor='#29B6F6', edgecolor='k', label='Snow (1)'),
        Patch(facecolor='#4E342E', edgecolor='k', label='No-Snow Ground (0)'),
        Patch(facecolor='#D6D6D6', edgecolor='k', label='Cloud / Masked (255)'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=9.5, bbox_to_anchor=(0.5, -0.06))

    save_all(fig, "multitemporal_winter_series")


if __name__ == "__main__":
    data_dir = r"C:\Users\rnine\LIS\data"
    plot_snowline_domain_shift(data_dir)
    plot_elevation_performance()
    plot_scene_cloud_and_snow_stats(data_dir)
    plot_sar_backscatter_distributions(data_dir)
    plot_precision_recall_comparison()
    plot_threshold_sensitivity()
    plot_s2_optical_lis_demo(data_dir)
    plot_multitemporal_winter_series(data_dir)
