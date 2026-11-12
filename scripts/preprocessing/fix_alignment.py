import os
import sys
import rasterio
from rasterio.warp import reproject, Resampling


def fix_alignment(data_dir):
    s1_path = os.path.join(data_dir, 'S1A_Terrain_Corrected.tif')
    lbl_path = os.path.join(data_dir, 'training_labels.tif')
    s1_fixed_path = os.path.join(data_dir, 'S1A_Terrain_Corrected_Fixed.tif')

    if not os.path.exists(s1_path) or not os.path.exists(lbl_path):
        print(f"Skipping {data_dir} - files not found.")
        return False

    print(f"Fixing alignment for {data_dir}...")

    with rasterio.open(s1_path) as src_s1, rasterio.open(lbl_path) as src_lbl:
        if src_s1.width == src_lbl.width and src_s1.height == src_lbl.height and src_s1.transform == src_lbl.transform:
            print("Already perfectly aligned. Skipping.")
            return True

        kwargs = src_s1.meta.copy()
        kwargs.update({
            'crs': src_lbl.crs,
            'transform': src_lbl.transform,
            'width': src_lbl.width,
            'height': src_lbl.height
        })

        with rasterio.open(s1_fixed_path, 'w', **kwargs) as dst:
            for i in range(1, src_s1.count + 1):
                reproject(
                    source=rasterio.band(src_s1, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src_s1.transform,
                    src_crs=src_s1.crs,
                    dst_transform=src_lbl.transform,
                    dst_crs=src_lbl.crs,
                    resampling=Resampling.nearest
                )

    os.replace(s1_fixed_path, s1_path)
    print("Fixed and replaced successfully!")
    return True


if __name__ == "__main__":
    base_dir = r'C:\Users\rnine\LIS\data'
    target = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(base_dir, 'patch_test')

    sub_patches = sorted(
        [d.path for d in os.scandir(target) if d.is_dir() and d.name.startswith("patch")],
        key=lambda p: int(p.split("patch")[-1]) if p.split("patch")[-1].isdigit() else 999
    )
    if sub_patches:
        for sp in sub_patches:
            fix_alignment(sp)
    else:
        fix_alignment(target)
