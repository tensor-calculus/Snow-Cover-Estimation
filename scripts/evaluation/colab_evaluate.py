import os
import sys
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))
from colab_train_nomle import BaselineUNet, compute_briand_metrics


def create_aligned_dem(dem_mosaic, reference_tif, output_dem):
    if not os.path.exists(dem_mosaic):
        return False

    print(f"Reprojecting {dem_mosaic} to match Sentinel-1 (10m UTM)...")
    with rasterio.open(dem_mosaic) as src, rasterio.open(reference_tif) as ref:
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': ref.crs,
            'transform': ref.transform,
            'width': ref.width,
            'height': ref.height,
            'dtype': rasterio.float32,
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'compress': 'deflate'
        })
        with rasterio.open(output_dem, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    resampling=Resampling.bilinear
                )
    return True


def make_gaussian_kernel(size=256, sigma=64.0):
    """2D Gaussian kernel giving higher weight to central pixels (Briand et al. Section 4.1)."""
    ax = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.max(kernel)


def compute_dem_features(dem_patch):
    """
    Computes 2 topographic channels from a 256x256 10m DEM patch:
      1. Normalized Elevation:dem_m / 6000.0
      2. Normalized Slope: arctan(|grad(Z)|) / (pi/2) in [0, 1]
    """
    dem_clean = np.nan_to_num(dem_patch, nan=0.0)
    dy, dx = np.gradient(dem_clean, 10.0)
    slope_rad = np.arctan(np.hypot(dx, dy))
    elev_norm = np.clip(dem_clean / 6000.0, 0.0, 1.5).astype(np.float32)
    slope_norm = (slope_rad / (np.pi / 2.0)).astype(np.float32)
    return elev_norm, slope_norm


def resolve_test_paths(root_test_dir):
    """Handles both flat patch_test/ and nested patch_test/winter_img/ structures."""
    winter_dir = os.path.join(root_test_dir, "winter_img")
    if os.path.isdir(winter_dir) and os.path.exists(os.path.join(winter_dir, "S1A_Terrain_Corrected.tif")):
        s1_dir = winter_dir
    else:
        s1_dir = root_test_dir

    s1_path = os.path.join(s1_dir, "S1A_Terrain_Corrected.tif")
    lbl_path = os.path.join(s1_dir, "training_labels.tif")

    dem_aligned_path = os.path.join(s1_dir, "DEM_aligned.tif")
    if not os.path.exists(dem_aligned_path):
        alt_aligned = os.path.join(root_test_dir, "DEM_aligned.tif")
        if os.path.exists(alt_aligned):
            dem_aligned_path = alt_aligned

    dem_mosaic_candidates = [
        os.path.join(s1_dir, "DEM_mosaic.tif"),
        os.path.join(root_test_dir, "DEM_mosaic.tif"),
        os.path.join(root_test_dir, "patch1", "DEM_mosaic.tif"),
    ]
    dem_mosaic_path = next((p for p in dem_mosaic_candidates if os.path.exists(p)), dem_mosaic_candidates[0])

    mle_candidates = [
        os.path.join(s1_dir, "S1A_MLE_Reference.tif"),
        os.path.join(root_test_dir, "S1A_MLE_Reference.tif"),
    ]
    mle_path = next((p for p in mle_candidates if os.path.exists(p)), mle_candidates[0])

    return s1_path, lbl_path, dem_aligned_path, dem_mosaic_path, mle_path


def evaluate_test_set():
    if os.name == 'nt':
        root_test_dir = r"C:\Users\rnine\LIS\data\patch_test"
        default_ckpt = r"C:\Users\rnine\LIS\models\with_mle.pth"
    else:
        root_test_dir = r"/content/drive/MyDrive/LIS_Data/patch_test"
        default_ckpt = r"/content/drive/MyDrive/LIS_Data/checkpoint_epoch_15.pth"

    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else default_ckpt
    s1_path, lbl_path, dem_aligned_path, dem_mosaic_path, mle_path = resolve_test_paths(root_test_dir)

    has_dem = os.path.exists(dem_aligned_path) or create_aligned_dem(dem_mosaic_path, lbl_path, dem_aligned_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Model from {checkpoint_path} onto {device}...")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    in_channels = checkpoint.get('in_channels', 2)
    fixed_threshold = float(checkpoint.get('best_threshold', 0.50))

    model = BaselineUNet(in_channels=in_channels).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if in_channels >= 4 and not os.path.exists(mle_path):
        raise FileNotFoundError(f"Model with {in_channels} channels requires {mle_path} for test evaluation.")
    if in_channels == 6 and not has_dem:
        raise FileNotFoundError(f"6-channel DEM model requires {dem_aligned_path} for test evaluation.")

    PATCH = 256
    STRIDE = 128
    gauss_w = make_gaussian_kernel(PATCH, sigma=64.0)

    with rasterio.open(s1_path) as src_s1, rasterio.open(lbl_path) as src_lbl:
        width, height = src_lbl.width, src_lbl.height
        src_mle = rasterio.open(mle_path) if in_channels >= 4 else None
        src_dem_in = rasterio.open(dem_aligned_path) if in_channels == 6 else None

        prob_sum = np.zeros((height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)

        print(
            f"Running Gaussian-weighted overlapping inference "
            f"(stride={STRIDE}, in_channels={in_channels}, fixed_val_th={fixed_threshold:.2f})..."
        )
        with torch.no_grad():
            for y in range(0, height - PATCH + 1, STRIDE):
                for x in range(0, width - PATCH + 1, STRIDE):
                    window = Window(x, y, PATCH, PATCH)
                    lbl_w = src_lbl.read(1, window=window)
                    if np.all(lbl_w == 255):
                        continue

                    s1 = src_s1.read([1, 2], window=window).astype(np.float32)
                    nodata = ~np.isfinite(s1).all(0) | (s1 <= 0).any(0)
                    if np.all(nodata):
                        continue

                    vh, vv = np.clip(np.nan_to_num(s1, nan=1e-5), 1e-5, None)
                    vh_db, vv_db = 10.0 * np.log10(vh), 10.0 * np.log10(vv)

                    if in_channels == 2:
                        img = np.stack([(vv_db + 30.0) / 30.0, (vh_db + 35.0) / 35.0], axis=0)
                    elif in_channels == 4:
                        mle = src_mle.read([1, 2], window=window).astype(np.float32)
                        nodata |= (~np.isfinite(mle).all(0) | (mle <= 0).any(0))
                        vh_ref, vv_ref = np.clip(np.nan_to_num(mle, nan=1e-5), 1e-5, None)
                        vh_ref_db, vv_ref_db = 10.0 * np.log10(vh_ref), 10.0 * np.log10(vv_ref)
                        img = np.stack([
                            (vv_db + 30.0) / 30.0,
                            (vh_db + 35.0) / 35.0,
                            (vv_ref_db + 30.0) / 30.0,
                            (vh_ref_db + 35.0) / 35.0
                        ], axis=0)
                    elif in_channels == 6:
                        mle = src_mle.read([1, 2], window=window).astype(np.float32)
                        dem_p = src_dem_in.read(1, window=window).astype(np.float32)
                        nodata |= (~np.isfinite(mle).all(0) | (mle <= 0).any(0) | ~np.isfinite(dem_p) | (dem_p <= 0))
                        vh_ref, vv_ref = np.clip(np.nan_to_num(mle, nan=1e-5), 1e-5, None)
                        vh_ref_db, vv_ref_db = 10.0 * np.log10(vh_ref), 10.0 * np.log10(vv_ref)
                        elev_norm, slope_norm = compute_dem_features(dem_p)
                        img = np.stack([
                            (vv_db + 30.0) / 30.0,
                            (vh_db + 35.0) / 35.0,
                            (vv_ref_db + 30.0) / 30.0,
                            (vh_ref_db + 35.0) / 35.0,
                            elev_norm,
                            slope_norm
                        ], axis=0)

                    img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)
                    if device.type == 'cuda':
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            logits = model(img_tensor)
                    else:
                        logits = model(img_tensor)

                    prob_snow = torch.softmax(logits.float(), dim=1)[0, 1].cpu().numpy()
                    valid_px = ~nodata
                    w_patch = gauss_w * valid_px

                    prob_sum[y:y+PATCH, x:x+PATCH] += prob_snow * w_patch
                    weight_sum[y:y+PATCH, x:x+PATCH] += w_patch

        if src_mle is not None:
            src_mle.close()
        if src_dem_in is not None:
            src_dem_in.close()

        full_lbl = src_lbl.read(1)
        valid_mask = (full_lbl != 255) & (full_lbl != 2) & (weight_sum > 0)
        mean_prob = np.zeros_like(prob_sum)
        mean_prob[valid_mask] = prob_sum[valid_mask] / weight_sum[valid_mask]

    bins = [0, 3000, 4000, 5000, 9000]
    bin_names = ["<3000m", "3000-4000m", "4000-5000m", ">5000m"]

    preds = (mean_prob >= fixed_threshold).astype(np.uint8)
    p_valid = preds[valid_mask]
    t_valid = full_lbl[valid_mask]

    print(f"\n================ Fixed Training/Validation Threshold (th={fixed_threshold:.2f}) ================")
    if has_dem:
        with rasterio.open(dem_aligned_path) as src_dem:
            dem = src_dem.read(1)[valid_mask]
        digitized = np.digitize(dem, bins) - 1

        for i, b_name in enumerate(bin_names):
            idx = (digitized == i)
            if np.any(idx):
                pb, tb = p_valid[idx], t_valid[idx]
                tp = float(np.sum((pb == 1) & (tb == 1)))
                fp = float(np.sum((pb == 1) & (tb == 0)))
                fn = float(np.sum((pb == 0) & (tb == 1)))
                tn = float(np.sum((pb == 0) & (tb == 0)))
                m = compute_briand_metrics(tp, fp, fn, tn)
                print(
                    f"[{b_name:10s}] Acc: {m['acc']*100:5.2f}% | Overall F1: {m['overall_f1']*100:5.2f}% | "
                    f"F1_Snow: {m['f1_snow']*100:5.2f}% | F1_NoSnow: {m['f1_nosnow']*100:5.2f}% | IoU: {m['iou_snow']:.4f}"
                )

    tp = float(np.sum((p_valid == 1) & (t_valid == 1)))
    fp = float(np.sum((p_valid == 1) & (t_valid == 0)))
    fn = float(np.sum((p_valid == 0) & (t_valid == 1)))
    tn = float(np.sum((p_valid == 0) & (t_valid == 0)))
    gm = compute_briand_metrics(tp, fp, fn, tn)

    print(f"--- GLOBAL METRICS (th={fixed_threshold:.2f}) ---")
    print(f"Overall Accuracy : {gm['acc']*100:.2f}%")
    print(f"Overall F1 (Eq22): {gm['overall_f1']*100:.2f}%  <-- Briand et al. main metric")
    print(f"F1-Snow          : {gm['f1_snow']*100:.2f}%")
    print(f"F1-NoSnow        : {gm['f1_nosnow']*100:.2f}%")
    print(f"Snow Precision   : {gm['prec_snow']*100:.2f}%")
    print(f"Snow Recall      : {gm['rec_snow']*100:.2f}%")
    print(f"Snow IoU         : {gm['iou_snow']:.4f}")


if __name__ == "__main__":
    evaluate_test_set()
