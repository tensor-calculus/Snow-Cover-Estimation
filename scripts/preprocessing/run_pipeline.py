import os
import sys
import glob
import zipfile
import subprocess
import json
from pathlib import Path
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import xml.etree.ElementTree as ET
import geopandas as gpd
from shapely.geometry import Polygon


def create_dem_mosaic(dem_dir, output_dem):
    print("Finding DEM tifs...")
    dem_files = glob.glob(os.path.join(dem_dir, "**", "DEM", "*_DEM.tif"), recursive=True)
    if not dem_files:
        print("No DEM files found.")
        return
    print(f"Found {len(dem_files)} DEM files. Mosaicking...")

    src_files_to_mosaic = [rasterio.open(fp) for fp in dem_files]
    mosaic, out_trans = merge(src_files_to_mosaic)
    out_meta = src_files_to_mosaic[0].meta.copy()

    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })

    with rasterio.open(output_dem, "w", **out_meta) as dest:
        dest.write(mosaic)

    for src in src_files_to_mosaic:
        src.close()
    print(f"Created DEM mosaic at {output_dem}")


def resolve_shared_file(data_dir, filename):
    """Finds DEM_mosaic.tif or patch_footprint.kml in data_dir, parent dir, or sibling patch1 without manual copying."""
    parent_dir = os.path.dirname(data_dir.rstrip("\\/"))
    candidates = [
        os.path.join(data_dir, filename),
        os.path.join(parent_dir, filename),
        os.path.join(parent_dir, "patch1", filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(data_dir, filename)


def auto_extract_s2_zips(data_dir, s2_extracted_dir):
    """Automatically extracts any S2*.zip files in data_dir into sentinel2_extracted/ if not already extracted."""
    s2_zips = glob.glob(os.path.join(data_dir, "S2*.zip"))
    if not s2_zips:
        return
    os.makedirs(s2_extracted_dir, exist_ok=True)
    existing_safes = [f.name for f in os.scandir(s2_extracted_dir) if f.is_dir() and f.name.endswith('.SAFE')]
    if len(existing_safes) >= len(s2_zips):
        return
    for zp in s2_zips:
        print(f"Auto-extracting Sentinel-2 archive: {os.path.basename(zp)}...")
        with zipfile.ZipFile(zp, 'r') as zf:
            zf.extractall(s2_extracted_dir)


def prepare_config(template_path, output_path, dem_path):
    with open(template_path, 'r') as f:
        config = json.load(f)

    config['general']['preprocessing'] = True
    config['shaded_snow']['shaded_snow_hillshade'] = False
    config['shaded_snow']['shaded_snow_casted'] = False

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Prepared config at {output_path}")


def run_let_it_snow(s2_extracted_dir, config_path, dem_path, output_dir):
    s2_folders = [f.path for f in os.scandir(s2_extracted_dir) if f.is_dir() and f.name.endswith('.SAFE')]
    os.makedirs(output_dir, exist_ok=True)

    for s2_folder in s2_folders:
        folder_name = os.path.basename(s2_folder)
        out_sub = os.path.join(output_dir, folder_name)
        os.makedirs(out_sub, exist_ok=True)

        def to_docker_path(p):
            return p.replace("C:\\Users\\rnine\\LIS", "/LIS").replace("\\", "/")

        d_in = to_docker_path(s2_folder)
        d_out = to_docker_path(out_sub)
        d_config = to_docker_path(config_path)
        d_dem = to_docker_path(dem_path)

        cmd = [
            "docker", "run", "--rm",
            "-v", "C:\\Users\\rnine\\LIS:/LIS",
            "lis", "python3", "/usr/local/app/let_it_snow_fsc.py",
            "-i", d_in,
            "-o", d_out,
            "-c", d_config,
            "-d", d_dem
        ]

        print(f"Running LIS for {folder_name}...")
        print("Command:", " ".join(cmd))
        subprocess.run(cmd, check=True)


def parse_kml_polygon(kml_path):
    tree = ET.parse(kml_path)
    root = tree.getroot()
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    coords_text = root.find('.//kml:coordinates', ns).text.strip()
    coords = []
    for pt in coords_text.split():
        parts = pt.split(',')
        coords.append((float(parts[0]), float(parts[1])))
    return Polygon(coords)


def mosaic_and_crop_labels(lis_output_dir, kml_path, final_output):
    print("Mosaicking output FSC masks to create binary labels...")
    fsc_files = glob.glob(os.path.join(lis_output_dir, "**", "*FSC-TOC*.tif"), recursive=True)
    if not fsc_files:
        fsc_files = glob.glob(os.path.join(lis_output_dir, "**", "*FSC*.tif"), recursive=True)

    fsc_files = [f for f in fsc_files if "QCFLAGS" not in f]

    if not fsc_files:
        print("No FSC files found to merge.")
        return

    src_files_to_mosaic = [rasterio.open(fp) for fp in fsc_files]
    mosaic, out_trans = merge(src_files_to_mosaic)

    import numpy as np
    dl_mosaic = np.copy(mosaic)
    dl_mosaic[(mosaic > 0) & (mosaic <= 100)] = 1
    dl_mosaic[mosaic == 205] = 2
    dl_mosaic[mosaic == 255] = 255

    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "nodata": 255
    })

    merged_path = final_output.replace(".tif", "_merged.tif")
    with rasterio.open(merged_path, "w", **out_meta) as dest:
        dest.write(dl_mosaic)
    for src in src_files_to_mosaic:
        src.close()

    print("Cropping and resampling to 10m...")
    poly = parse_kml_polygon(kml_path)
    gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[poly])

    with rasterio.open(merged_path) as src:
        gdf = gdf.to_crs(src.crs)
        bbox = gdf.total_bounds

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, src.crs, src.width, src.height,
            left=bbox[0], bottom=bbox[1], right=bbox[2], top=bbox[3],
            resolution=(10, 10)
        )

        dst_kwargs = src.meta.copy()
        dst_kwargs.update({
            'crs': src.crs,
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height,
            'nodata': 255
        })

        with rasterio.open(final_output, 'w', **dst_kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=src.crs,
                    resampling=Resampling.nearest
                )
    print(f"Final binary training labels saved to {final_output}")


def run_for_patch(data_dir):
    base_dir = r"C:\Users\rnine\LIS"
    print(f"\n==========================================")
    print(f"Running LIS pipeline for directory: {data_dir}")
    print(f"==========================================")

    final_labels = os.path.join(data_dir, "training_labels.tif")
    if os.path.exists(final_labels):
        print(f"training_labels.tif already exists in {data_dir}. Skipping LIS.")
        return

    dem_extracted = os.path.join(data_dir, "DEM_extracted")
    dem_mosaic = resolve_shared_file(data_dir, "DEM_mosaic.tif")
    kml_footprint = resolve_shared_file(data_dir, "patch_footprint.kml")

    s2_extracted = os.path.join(data_dir, "sentinel2_extracted")
    auto_extract_s2_zips(data_dir, s2_extracted)

    lis_output = os.path.join(data_dir, "lis_output")
    config_template = os.path.join(base_dir, "let-it-snow", "doc", "lis_default_configuration.json")
    config_out = os.path.join(data_dir, "lis_config.json")

    if not os.path.exists(dem_mosaic):
        create_dem_mosaic(dem_extracted, dem_mosaic)

    prepare_config(config_template, config_out, dem_mosaic)

    print("Checking if 'lis' docker image exists...")
    res = subprocess.run(["docker", "images", "-q", "lis"], capture_output=True, text=True)
    if not res.stdout.strip():
        print("Docker image 'lis' not found. Building it...")
        subprocess.run(["docker", "build", "-t", "lis", "."], cwd=os.path.join(base_dir, "let-it-snow"), check=True)

    run_let_it_snow(s2_extracted, config_out, dem_mosaic, lis_output)
    mosaic_and_crop_labels(lis_output, kml_footprint, final_labels)


if __name__ == "__main__":
    base_dir = r"C:\Users\rnine\LIS"
    target = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(base_dir, "data", "patch1")

    sub_patches = sorted(
        [d.path for d in os.scandir(target) if d.is_dir() and d.name.startswith("patch")],
        key=lambda p: int(p.split("patch")[-1]) if p.split("patch")[-1].isdigit() else 999
    )
    if sub_patches:
        for sp in sub_patches:
            run_for_patch(sp)
    else:
        run_for_patch(target)
