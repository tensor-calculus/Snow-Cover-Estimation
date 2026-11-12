import os
import sys
import rasterio


def convert_to_tiled(filepath):
    if not os.path.exists(filepath):
        return
    with rasterio.open(filepath) as src:
        if src.profile.get('tiled', False):
            print(f"{filepath} is already tiled. Skipping.")
            return

    temp_path = filepath.replace('.tif', '_temp.tif')
    os.rename(filepath, temp_path)

    with rasterio.open(temp_path) as src:
        kwargs = src.meta.copy()
        kwargs.update({
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'compress': 'deflate',
            'interleave': 'band'
        })

        print(f"Tiling {filepath}...")
        with rasterio.open(filepath, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                dst.write(src.read(i), i)

    os.remove(temp_path)


if __name__ == "__main__":
    base_dir = r"C:\Users\rnine\LIS\data"
    target = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(base_dir, "patch_test")

    sub_patches = sorted(
        [d.path for d in os.scandir(target) if d.is_dir() and d.name.startswith("patch")],
        key=lambda p: int(p.split("patch")[-1]) if p.split("patch")[-1].isdigit() else 999
    )
    if sub_patches:
        for sp in sub_patches:
            convert_to_tiled(os.path.join(sp, 'S1A_Terrain_Corrected.tif'))
            convert_to_tiled(os.path.join(sp, 'training_labels.tif'))
            print(f"Finished tiling {sp}")
    else:
        convert_to_tiled(os.path.join(target, 'S1A_Terrain_Corrected.tif'))
        convert_to_tiled(os.path.join(target, 'training_labels.tif'))
        print(f"Finished tiling {target}")

    print("\nAll files successfully converted to tiled format!")
