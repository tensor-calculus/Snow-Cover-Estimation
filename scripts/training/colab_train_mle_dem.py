import os
import glob
import random
import re
import shutil
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.amp


# --- 0. ENSURE TILED MLE & ALIGNED 10m DEM EXIST ---
def ensure_mle_and_dem(data_dir, ckpt_dir=None):
    """
    Ensures S1A_MLE_Reference.tif and DEM_aligned.tif exist in data_dir.
    If DEM_aligned.tif is missing, reprojects DEM_mosaic.tif to match patch7's 10m UTM grid
    with 256x256 internal tiling.
    """
    mle_path = os.path.join(data_dir, "S1A_MLE_Reference.tif")
    dem_aligned_path = os.path.join(data_dir, "DEM_aligned.tif")
    drive_root = os.path.dirname(ckpt_dir.rstrip("/\\")) if ckpt_dir else None

    if not os.path.exists(mle_path) and drive_root:
        drive_mle = os.path.join(drive_root, "S1A_MLE_Reference.tif")
        if os.path.exists(drive_mle):
            print(f"Copying {drive_mle} to {mle_path}...")
            shutil.copy2(drive_mle, mle_path)

    if not os.path.exists(dem_aligned_path) and drive_root:
        drive_dem_aligned = os.path.join(drive_root, "DEM_aligned.tif")
        if os.path.exists(drive_dem_aligned):
            print(f"Copying {drive_dem_aligned} to {dem_aligned_path}...")
            shutil.copy2(drive_dem_aligned, dem_aligned_path)

    if not os.path.exists(dem_aligned_path):
        dem_mosaic = os.path.join(data_dir, "DEM_mosaic.tif")
        if not os.path.exists(dem_mosaic) and drive_root:
            drive_mosaic = os.path.join(drive_root, "DEM_mosaic.tif")
            if os.path.exists(drive_mosaic):
                dem_mosaic = drive_mosaic

        ref_tif = os.path.join(data_dir, "patch7", "training_labels.tif")
        if not os.path.exists(dem_mosaic):
            raise FileNotFoundError(
                f"Missing {dem_aligned_path} and {dem_mosaic}. "
                "Upload DEM_aligned.tif (or DEM_mosaic.tif) to LIS_Data!"
            )

        print(f"Reprojecting and tiling {dem_mosaic} -> {dem_aligned_path} (10m UTM, 256x256 blocks)...")
        with rasterio.open(dem_mosaic) as src, rasterio.open(ref_tif) as ref:
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': ref.crs,
                'transform': ref.transform,
                'width': ref.width,
                'height': ref.height,
                'dtype': 'float32',
                'count': 1,
                'tiled': True,
                'blockxsize': 256,
                'blockysize': 256,
                'compress': 'deflate',
                'BIGTIFF': 'YES'
            })
            with rasterio.open(dem_aligned_path, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    resampling=Resampling.bilinear
                )
        print(f"Successfully created {dem_aligned_path}")
        if drive_root and os.path.isdir(drive_root):
            drive_dem_aligned = os.path.join(drive_root, "DEM_aligned.tif")
            if not os.path.exists(drive_dem_aligned):
                print(f"Caching {dem_aligned_path} to {drive_dem_aligned}...")
                shutil.copy2(dem_aligned_path, drive_dem_aligned)

    return mle_path, dem_aligned_path


def compute_dem_features(dem_patch):
    """
    Computes 2 topographic channels from a 256x256 10m DEM patch:
      1. Normalized Elevation: dem_m / 6000.0
      2. Normalized Slope: arctan(|grad(Z)|) / (pi/2) in [0, 1]
    """
    dem_clean = np.nan_to_num(dem_patch, nan=0.0)
    dy, dx = np.gradient(dem_clean, 10.0)
    slope_rad = np.arctan(np.hypot(dx, dy))
    elev_norm = np.clip(dem_clean / 6000.0, 0.0, 1.5).astype(np.float32)
    slope_norm = (slope_rad / (np.pi / 2.0)).astype(np.float32)
    return elev_norm, slope_norm


# --- 1. DATASET DEFINITION (6 CHANNELS: VV, VH, VV_ref, VH_ref, Elevation, Slope) ---
class ColabSnowMLEDEMDataset(Dataset):
    def __init__(self, patch_dirs, mle_path, dem_path):
        self.samples = []
        self.mle_path = mle_path
        self.dem_path = dem_path
        self.handles = None
        if not os.path.exists(mle_path):
            raise FileNotFoundError(f"Missing MLE Reference file: {mle_path}")
        if not os.path.exists(dem_path):
            raise FileNotFoundError(f"Missing DEM file: {dem_path}")

        for p_dir in patch_dirs:
            coords_path = os.path.join(p_dir, 'valid_coords.npy')
            s1_path = os.path.join(p_dir, 'S1A_Terrain_Corrected.tif')
            lbl_path = os.path.join(p_dir, 'training_labels.tif')
            if not os.path.exists(coords_path):
                raise FileNotFoundError(f"Missing coordinates file: {coords_path}")
            coords = np.load(coords_path)
            for x, y in coords:
                self.samples.append({'s1_path': s1_path, 'lbl_path': lbl_path, 'x': int(x), 'y': int(y)})

    def __len__(self):
        return len(self.samples)

    def _get_handle(self, path):
        if self.handles is None:
            self.handles = {}
        if path not in self.handles:
            self.handles[path] = rasterio.open(path)
        return self.handles[path]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        window = Window(sample['x'], sample['y'], 256, 256)

        src_s1 = self._get_handle(sample['s1_path'])
        src_lbl = self._get_handle(sample['lbl_path'])
        src_mle = self._get_handle(self.mle_path)
        src_dem = self._get_handle(self.dem_path)

        # Read winter S1, Summer MLE Reference (Band 1 = VH, Band 2 = VV), and 10m DEM
        s1 = src_s1.read([1, 2], window=window).astype(np.float32)
        mle = src_mle.read([1, 2], window=window).astype(np.float32)
        dem = src_dem.read(1, window=window).astype(np.float32)
        lbl = src_lbl.read(1, window=window)

        # Mask out areas with no winter radar data, no valid snow-free reference (e.g. permasnow), or no DEM
        nodata = (
            ~np.isfinite(s1).all(0) | (s1 <= 0).any(0) |
            ~np.isfinite(mle).all(0) | (mle <= 0).any(0) |
            ~np.isfinite(dem) | (dem <= 0)
        )

        vh, vv = np.clip(np.nan_to_num(s1, nan=1e-5), 1e-5, None)
        vh_ref, vv_ref = np.clip(np.nan_to_num(mle, nan=1e-5), 1e-5, None)

        vh_db, vv_db = 10.0 * np.log10(vh), 10.0 * np.log10(vv)
        vh_ref_db, vv_ref_db = 10.0 * np.log10(vh_ref), 10.0 * np.log10(vv_ref)

        elev_norm, slope_norm = compute_dem_features(dem)

        # Concatenate [VV, VH, VV_ref, VH_ref, Elevation, Slope]
        img = np.stack([
            (vv_db + 30.0) / 30.0,
            (vh_db + 35.0) / 35.0,
            (vv_ref_db + 30.0) / 30.0,
            (vh_ref_db + 35.0) / 35.0,
            elev_norm,
            slope_norm
        ], axis=0)

        lbl[lbl == 2] = 255
        lbl[nodata] = 255

        return torch.from_numpy(img).float(), torch.from_numpy(lbl).long()


# --- 2. U-NET ARCHITECTURE ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class BaselineUNet(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        y1 = self.conv1(torch.cat([x2, self.up1(x3)], dim=1))
        y2 = self.conv2(torch.cat([x1, self.up2(y1)], dim=1))
        return self.outc(y2)


# --- 3. STEP 1: BRIAND ET AL. METRICS & THRESHOLD SWEEP ---
@torch.no_grad()
def threshold_confusion_counts(logits, targets, thresholds):
    """
    Accumulates [TP, FP, FN, TN] across multiple probability thresholds in one pass
    for Briand et al. threshold optimization and exact metric calculation.
    """
    probs = torch.softmax(logits.float(), dim=1)[:, 1]
    m = targets != 255
    if not m.any():
        return torch.zeros((len(thresholds), 4), device=logits.device, dtype=torch.long)

    p = probs[m].unsqueeze(0)              # (1, N)
    t = (targets[m] == 1).unsqueeze(0)     # (1, N)
    preds = p >= thresholds.unsqueeze(1)   # (T, N)

    tp = (preds & t).sum(dim=1)
    fp = (preds & (~t)).sum(dim=1)
    fn = ((~preds) & t).sum(dim=1)
    tn = ((~preds) & (~t)).sum(dim=1)
    return torch.stack([tp, fp, fn, tn], dim=1)


def compute_briand_metrics(tp, fp, fn, tn):
    """
    Computes exact Briand et al. (2026) Section 4.4.1 metrics (Eq. 21 & Eq. 22)
    plus Snow IoU.
    """
    total = tp + fp + fn + tn + 1e-8
    f_snow = (tp + fn) / total

    acc = (tp + tn) / total
    prec_snow = tp / (tp + fp + 1e-8)
    rec_snow = tp / (tp + fn + 1e-8)
    f1_snow = 2.0 * tp / (2.0 * tp + fp + fn + 1e-8)
    iou_snow = tp / (tp + fp + fn + 1e-8)

    prec_nosnow = tn / (tn + fn + 1e-8)
    rec_nosnow = tn / (tn + fp + 1e-8)
    f1_nosnow = 2.0 * tn / (2.0 * tn + fp + fn + 1e-8)

    # Briand Eq. 22: frequency-weighted Overall F1
    overall_f1 = f_snow * f1_snow + (1.0 - f_snow) * f1_nosnow

    return {
        'acc': float(acc),
        'overall_f1': float(overall_f1),
        'f1_snow': float(f1_snow),
        'f1_nosnow': float(f1_nosnow),
        'iou_snow': float(iou_snow),
        'prec_snow': float(prec_snow),
        'rec_snow': float(rec_snow)
    }


# --- 4. HIGH-PERFORMANCE TRAINING LOOP ---
def train():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    if os.name == 'nt':
        data_dir = r"C:\Users\rnine\LIS\data"
        ckpt_dir = data_dir
    else:
        if not os.path.isdir("/content/drive/MyDrive"):
            raise RuntimeError("Drive not mounted! Mount Drive via UI or drive.mount('/content/drive') first.")
        data_dir = "/content/data"
        ckpt_dir = "/content/drive/MyDrive/LIS_Data/checkpoints_mle_dem"

    mle_path, dem_path = ensure_mle_and_dem(data_dir, ckpt_dir)

    train_dirs = [os.path.join(data_dir, f"patch{i}") for i in range(7, 12)]
    val_dirs = [os.path.join(data_dir, "patch12")]

    train_dataset = ColabSnowMLEDEMDataset(train_dirs, mle_path, dem_path)
    val_dataset = ColabSnowMLEDEMDataset(val_dirs, mle_path, dem_path)

    # Fast validation: Non-overlapping windows (stride=256) across full patch12
    val_dataset.samples = [
        s for s in val_dataset.samples
        if s['x'] % 256 == 0 and s['y'] % 256 == 0
    ]

    print(f"[WITH-MLE+DEM Channel Set C (6-ch)] Train Patches: {len(train_dataset)} | Val Patches: {len(val_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    g = torch.Generator()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=g,
                              num_workers=2, pin_memory=use_cuda, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                            num_workers=2, pin_memory=use_cuda, persistent_workers=True)

    model = BaselineUNet(in_channels=6).to(device)

    class_weights = torch.tensor([0.2, 0.8]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scaler = torch.amp.GradScaler(device.type, enabled=use_cuda)

    start_epoch = 0
    epochs = 15
    best_f1_snow = -1.0

    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoints = glob.glob(os.path.join(ckpt_dir, "checkpoint_mledem_epoch_*.pth"))
    if checkpoints:
        latest_ckpt = max(checkpoints, key=lambda p: int(re.search(r"epoch_(\d+)\.pth$", p).group(1)))
        print(f"Resuming from checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_f1_snow = checkpoint.get('best_f1_snow', -1.0)
        print(f"Starting at epoch {start_epoch + 1}")

    thresholds = torch.linspace(0.10, 0.90, 17, device=device)
    print(f"Starting Training on {device}...")

    for epoch in range(start_epoch, epochs):
        g.manual_seed(42 + epoch)

        model.train()
        for i, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_cuda):
                outputs = model(imgs)
                loss = criterion(outputs, lbls)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Validation Loop with Briand et al. Threshold Optimization
        model.eval()
        all_counts = torch.zeros((len(thresholds), 4), device=device, dtype=torch.long)
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_cuda):
                    outputs = model(imgs)
                all_counts += threshold_confusion_counts(outputs, lbls, thresholds)

        counts_cpu = all_counts.double().cpu().numpy()
        t_list = thresholds.cpu().numpy()

        # Default threshold = 0.50 (index 8)
        m_def = compute_briand_metrics(*counts_cpu[8])

        # Select optimal threshold maximizing F1_Snow (Briand Section 4.1)
        best_idx = 8
        best_th_m = m_def
        for idx in range(len(t_list)):
            m_cand = compute_briand_metrics(*counts_cpu[idx])
            if m_cand['f1_snow'] > best_th_m['f1_snow']:
                best_th_m = m_cand
                best_idx = idx
        best_th = float(t_list[best_idx])

        print(
            f"--> Val Epoch {epoch+1} [th=0.50] | Acc: {m_def['acc']*100:.2f}% | "
            f"Overall F1: {m_def['overall_f1']*100:.2f}% | F1_Snow: {m_def['f1_snow']*100:.2f}% | "
            f"F1_NoSnow: {m_def['f1_nosnow']*100:.2f}% | IoU: {m_def['iou_snow']:.4f}"
        )
        print(
            f"    Optimal Threshold [th={best_th:.2f}] | Acc: {best_th_m['acc']*100:.2f}% | "
            f"Overall F1: {best_th_m['overall_f1']*100:.2f}% | F1_Snow: {best_th_m['f1_snow']*100:.2f}% | "
            f"F1_NoSnow: {best_th_m['f1_nosnow']*100:.2f}% | IoU: {best_th_m['iou_snow']:.4f}"
        )

        if best_th_m['f1_snow'] > best_f1_snow:
            best_f1_snow = best_th_m['f1_snow']

        checkpoint_path = os.path.join(ckpt_dir, f"checkpoint_mledem_epoch_{epoch+1}.pth")
        tmp_path = checkpoint_path + ".tmp"

        torch.save({
            'epoch': epoch + 1,
            'in_channels': 6,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': loss.item(),
            'best_threshold': best_th,
            'val_metrics_050': m_def,
            'val_metrics_opt': best_th_m,
            'best_f1_snow': best_f1_snow
        }, tmp_path)

        os.replace(tmp_path, checkpoint_path)
        print(f"Saved Checkpoint to {checkpoint_path}")

    print("Training completely finished!")


if __name__ == "__main__":
    train()
