import os
import rasterio
import numpy as np
from rasterio.windows import Window
from rasterio.enums import Resampling

def setup_onthefly(data_dir):
    s1_path = os.path.join(data_dir, 'S1A_Terrain_Corrected.tif')
    lbl_path = os.path.join(data_dir, 'training_labels.tif')
    coords_path = os.path.join(data_dir, 'valid_coords.npy')
    
    if not os.path.exists(s1_path) or not os.path.exists(lbl_path):
        return
        
    print(f"Processing {data_dir}...")
    
    patch_size = 256
    stride = 128
    valid_coords = []
    
    print("  Scanning labels and S1 for valid data patches...")
    with rasterio.open(lbl_path) as src_lbl, rasterio.open(s1_path) as src_s1:
        width = src_lbl.width
        height = src_lbl.height
        
        full_lbl = src_lbl.read(1)
        
        # Low-res read of S1 to find valid data areas quickly and avoid training on 0-filled radar patches
        f = 8
        s1_lr = src_s1.read(out_shape=(2, src_s1.height // f, src_s1.width // f), resampling=Resampling.nearest)
        s1_valid = (s1_lr != 0).any(axis=0)
        
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                patch_lbl = full_lbl[y:y+patch_size, x:x+patch_size]
                
                # If it's NOT 100% NoData, AND has some valid S1 data, keep the coordinate
                if not np.all(patch_lbl == 255) and s1_valid[y//f:(y+patch_size)//f, x//f:(x+patch_size)//f].any():
                    valid_coords.append([x, y])
                    
    coords_array = np.array(valid_coords)
    np.save(coords_path, coords_array)
    print(f"  Found {len(coords_array)} valid patches. Saved coordinates.")

if __name__ == "__main__":
    base_dir = r"C:\Users\rnine\LIS\data"
    # for i in range(7, 13):
    #     setup_onthefly(os.path.join(base_dir, f"patch{i}"))
    # print("\nSetup complete! You can now run the on-the-fly training.")
    setup_onthefly(os.path.join(base_dir, f"patch_test"))
    print("\nSetup complete! You can now run the on-the-fly training.")