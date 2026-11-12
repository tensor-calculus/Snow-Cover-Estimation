import os
import rasterio
from rasterio.windows import Window
import numpy as np

def generate_mle_reference(data_dir):
    out_path = os.path.join(data_dir, "S1A_MLE_Reference.tif")
    summer_dirs = [
        os.path.join(data_dir, f"patch{i}") for i in range(1, 7)
        if os.path.exists(os.path.join(data_dir, f"patch{i}", "S1A_Terrain_Corrected.tif"))
    ]
    if not summer_dirs:
        raise FileNotFoundError(f"No processed summer patches (patch1..patch6) found in {data_dir}")
    
    s1_srcs = [rasterio.open(os.path.join(d, "S1A_Terrain_Corrected.tif")) for d in summer_dirs]
    lbl_srcs = [rasterio.open(os.path.join(d, "training_labels.tif")) for d in summer_dirs]
    
    height, width = s1_srcs[0].height, s1_srcs[0].width
    profile = s1_srcs[0].meta.copy()
    profile.update({
        'dtype': 'float32',
        'count': 2,
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'compress': 'deflate',
        'interleave': 'band',
        'BIGTIFF': 'YES'
    })
    
    print(f"Computing Rayleigh MLE Reference across {len(summer_dirs)} summer patches ({width}x{height})...")
    strip_h = 1024
    
    with rasterio.open(out_path, 'w', **profile) as dst:
        for y in range(0, height, strip_h):
            h = min(strip_h, height - y)
            win = Window(0, y, width, h)
            
            # Accumulators for strict clear-sky snow-free (lbl == 0)
            sum_sq_clear = np.zeros((2, h, width), dtype=np.float64)
            cnt_clear = np.zeros((h, width), dtype=np.int32)
            
            # Accumulators for cloud-penetrating SAR fallback (excluding permasnow lbl == 1)
            sum_sq_all = np.zeros((2, h, width), dtype=np.float64)
            cnt_all = np.zeros((h, width), dtype=np.int32)
            ever_permasnow = np.zeros((h, width), dtype=bool)
            
            for s1_src, lbl_src in zip(s1_srcs, lbl_srcs):
                s1 = s1_src.read([1, 2], window=win).astype(np.float32)
                lbl = lbl_src.read(1, window=win)
                
                valid_sar = np.isfinite(s1).all(axis=0) & (s1 > 0).all(axis=0) & (lbl != 255)
                ever_permasnow |= (lbl == 1)
                
                s1_sq = np.square(np.nan_to_num(s1, nan=0.0), dtype=np.float64)
                
                # 1. Confirmed optical snow-free & cloud-free (lbl == 0)
                m_clear = valid_sar & (lbl == 0)
                sum_sq_clear[:, m_clear] += s1_sq[:, m_clear]
                cnt_clear[m_clear] += 1
                
                # 2. All valid summer SAR observations (excluding snow lbl == 1)
                m_nosnow = valid_sar & (lbl != 1)
                sum_sq_all[:, m_nosnow] += s1_sq[:, m_nosnow]
                cnt_all[m_nosnow] += 1
                
            # Rayleigh MLE: sigma_ref = sqrt( 1 / (2*N) * sum(sigma^2) )
            mle_out = np.zeros((2, h, width), dtype=np.float32)
            
            # Primary: use confirmed clear-sky snow-free observations where available
            has_clear = cnt_clear > 0
            if np.any(has_clear):
                mle_out[:, has_clear] = np.sqrt(sum_sq_clear[:, has_clear] / (2.0 * cnt_clear[has_clear])).astype(np.float32)
                
            # Secondary: where summer optical was 100% cloudy all 6 dates and never showed permasnow, use SAR cloud-penetrating MLE
            fallback = (~has_clear) & (cnt_all > 0) & (~ever_permasnow)
            if np.any(fallback):
                mle_out[:, fallback] = np.sqrt(sum_sq_all[:, fallback] / (2.0 * cnt_all[fallback])).astype(np.float32)
                
            dst.write(mle_out, window=win)
            print(f"  Processed rows {y} to {y+h} / {height} ({(y+h)/height*100:.1f}%)")
            
    for src in s1_srcs + lbl_srcs:
        src.close()
        
    print(f"Successfully saved tiled MLE reference to: {out_path}")

if __name__ == "__main__":
    import sys
    target_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else r"C:\Users\rnine\LIS\data"
    generate_mle_reference(target_dir)
