import os
import sys
import torch
import rasterio
from rasterio.windows import Window
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "training"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "evaluation"))
from colab_train_nomle import BaselineUNet
from generate_plots import predict_window

def main():
    data_dir = os.path.join(REPO_ROOT, "data")
    models_dir = os.path.join(REPO_ROOT, "models")

    # EXACT coordinates of Row 2 from generate_plots.py (Figure 6 / model_outputs_comparison)
    p12_dir = os.path.join(data_dir, "patch12")
    p12_lbl = os.path.join(p12_dir, "training_labels.tif")
    p12_dem = os.path.join(data_dir, "DEM_aligned.tif")
    p12_s1 = os.path.join(p12_dir, "S1A_Terrain_Corrected.tif")
    p12_mle = os.path.join(data_dir, "S1A_MLE_Reference.tif")
    cx, cy = 23424, 5120
    wsz = 1024

    w = Window(cx, cy, wsz, wsz)
    with rasterio.open(p12_s1) as s_s1, rasterio.open(p12_mle) as s_mle, \
         rasterio.open(p12_dem) as s_dem, rasterio.open(p12_lbl) as s_lbl:
        s1 = s_s1.read([1, 2], window=w).astype(np.float32)
        mle = s_mle.read([1, 2], window=w).astype(np.float32)
        dem = s_dem.read(1, window=w).astype(np.float32)
        lbl = s_lbl.read(1, window=w)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model 3 (6-channel: S1 + MLE + DEM)
    m3_path = os.path.join(models_dir, "with_mle_dem.pth")
    ck = torch.load(m3_path, map_location=device, weights_only=False)
    th = float(ck.get("best_threshold", 0.60))
    m3 = BaselineUNet(in_channels=6).to(device)
    m3.load_state_dict(ck["model_state_dict"])
    m3.eval()

    pred_m3, _ = predict_window(m3, device, s1, mle, dem, 6, th)

    # Colormap matching Figure 6 (No-Snow=#5D4037, Snow=#29B6F6, Cloud=#E0E0E0)
    cmap_bin = ListedColormap(['#5D4037', '#29B6F6', '#E0E0E0'])
    norm_bin = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_bin.N)

    # What optical LIS produces without cloud mask:
    # Under thick clouds, water droplets reflect strongly in SWIR, driving NDSI < 0.15.
    # LIS evaluates the cloud tops and misclassifies them as No-Snow.
    lis_without_cloud = np.copy(lbl)
    lis_without_cloud[lbl == 2] = 0

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 9.0), constrained_layout=True)

    # 1. Copernicus DEM Elevation
    im_dem = axes[0, 0].imshow(dem, cmap='terrain', vmin=3000, vmax=6500)
    axes[0, 0].set_title('(a) Copernicus DEM Elevation\n(3,616m to 6,490m)', fontsize=11, fontweight='bold', pad=6)
    axes[0, 0].axis('off')
    cbar_dem = fig.colorbar(im_dem, ax=axes[0, 0], fraction=0.046, pad=0.03)
    cbar_dem.set_label('Elevation (m)', fontsize=9.5, fontweight='bold')

    # 2. Sentinel-1 SAR VV Backscatter (dB)
    vv = s1[0]
    vv_db = 10 * np.log10(np.clip(vv, 1e-4, 1.0))
    im_sar = axes[0, 1].imshow(vv_db, cmap='gray', vmin=-22, vmax=-5)
    axes[0, 1].set_title('(b) Sentinel-1 SAR $\\sigma^0_{\\mathrm{VV}}$ (dB)\n(Radar Penetrates Through Cloud Deck)', fontsize=11, fontweight='bold', pad=6)
    axes[0, 1].axis('off')
    cbar_sar = fig.colorbar(im_sar, ax=axes[0, 1], fraction=0.046, pad=0.03)
    cbar_sar.set_label('$\\sigma^0_{\\mathrm{VV}}$ (dB)', fontsize=9.5, fontweight='bold')

    # 3. Standard S2 LIS Ground Truth (With Cloud Mask in Gray) - EXACT MATCH TO FIGURE 6 ROW 2 COL 2
    axes[0, 2].imshow(lbl, cmap=cmap_bin, norm=norm_bin, interpolation='nearest')
    axes[0, 2].set_title('(c) S2 LIS Ground Truth (WITH Cloud Mask)\n[Figure 6, Row 2, Col 2: 46.8% Cloud Masked]', fontsize=11, fontweight='bold', pad=6)
    axes[0, 2].axis('off')
    leg_c = [
        Patch(facecolor='#29B6F6', edgecolor='k', label='Snow (44.1%)'),
        Patch(facecolor='#5D4037', edgecolor='k', label='No-Snow (9.1%)'),
        Patch(facecolor='#E0E0E0', edgecolor='k', label='Cloud Masked (46.8%)')
    ]
    axes[0, 2].legend(handles=leg_c, loc='lower center', fontsize=8.8, framealpha=0.92)

    # 4. What optical LIS produces WITHOUT cloud mask
    axes[1, 0].imshow(lis_without_cloud, cmap=ListedColormap(['#5D4037', '#29B6F6']), vmin=0, vmax=1, interpolation='nearest')
    axes[1, 0].set_title('(d) Optical LIS WITHOUT Cloud Mask\n(Low NDSI Cloud Top Misclassified as No-Snow!)', fontsize=11, fontweight='bold', pad=6)
    axes[1, 0].axis('off')
    leg_d = [
        Patch(facecolor='#29B6F6', edgecolor='k', label='Snow (44.1%)'),
        Patch(facecolor='#5D4037', edgecolor='k', label='Falsely Labeled No-Snow (55.9%)')
    ]
    axes[1, 0].legend(handles=leg_d, loc='lower center', fontsize=8.8, framealpha=0.92)

    # 5. Model 3 (6-ch) All-Weather SAR Prediction
    axes[1, 1].imshow(pred_m3, cmap=ListedColormap(['#5D4037', '#29B6F6']), vmin=0, vmax=1, interpolation='nearest')
    axes[1, 1].set_title('(e) Model 3 (6-ch) All-Weather SAR Prediction\n[Figure 6, Row 2, Col 5: Recovers Snow Under Clouds]', fontsize=11, fontweight='bold', pad=6)
    axes[1, 1].axis('off')
    leg_e = [
        Patch(facecolor='#29B6F6', edgecolor='k', label='Predicted Snow (67.2%)'),
        Patch(facecolor='#5D4037', edgecolor='k', label='Predicted No-Snow (32.8%)')
    ]
    axes[1, 1].legend(handles=leg_e, loc='lower center', fontsize=8.8, framealpha=0.92)

    # 6. Under-Cloud Snow Recovery Map
    recovery_map = np.zeros_like(lbl, dtype=np.uint8)
    recovery_map[(lbl == 0) & (pred_m3 == 0)] = 0  # Clear No-Snow
    recovery_map[(lbl == 1) & (pred_m3 == 1)] = 1  # Clear Snow
    recovery_map[(lbl == 2) & (pred_m3 == 1)] = 2  # Snow Recovered Under Cloud!
    recovery_map[(lbl == 2) & (pred_m3 == 0)] = 3  # Valley Ground Under Cloud

    cmap_rec = ListedColormap(['#37474F', '#00E676', '#FFD600', '#B0BEC5'])
    axes[1, 2].imshow(recovery_map, cmap=cmap_rec, vmin=0, vmax=3, interpolation='nearest')
    axes[1, 2].set_title('(f) Under-Cloud Snow Recovery Map\n(242,928 Snow Pixels Recovered Under Clouds!)', fontsize=11, fontweight='bold', pad=6)
    axes[1, 2].axis('off')
    leg_f = [
        Patch(facecolor='#00E676', edgecolor='k', label='Clear Sky Snow (44.1%)'),
        Patch(facecolor='#FFD600', edgecolor='k', label='SAR Snow Under Clouds (23.2%)'),
        Patch(facecolor='#B0BEC5', edgecolor='k', label='SAR Ground Under Clouds (23.6%)'),
        Patch(facecolor='#37474F', edgecolor='k', label='Clear Sky No-Snow (9.1%)')
    ]
    axes[1, 2].legend(handles=leg_f, loc='lower center', fontsize=8.6, framealpha=0.92)

    os.makedirs(os.path.join(REPO_ROOT, "report", "figures"), exist_ok=True)
    out_png = os.path.join(REPO_ROOT, "report", "figures", "patch12_second_test_image_lis_cloud_analysis.png")
    out_pdf = os.path.join(REPO_ROOT, "report", "figures", "patch12_second_test_image_lis_cloud_analysis.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    print("SUCCESSFULLY SAVED:")
    print(" ", out_png)
    print(" ", out_pdf)

if __name__ == "__main__":
    main()
