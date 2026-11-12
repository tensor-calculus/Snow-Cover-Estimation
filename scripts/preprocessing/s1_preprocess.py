import os
import subprocess
import glob
import zipfile
import shutil
import sys


def create_snap_graph():
    """Generates the SNAP XML graph for Sentinel-1 Preprocessing."""
    xml_content = """<graph id="Graph">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>${input_file}</file>
    </parameters>
  </node>
  <node id="Apply-Orbit-File">
    <operator>Apply-Orbit-File</operator>
    <sources>
      <sourceProduct refid="Read"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <orbitType>Sentinel Restituted (Auto Download)</orbitType>
      <polyDegree>3</polyDegree>
      <continueOnFail>false</continueOnFail>
    </parameters>
  </node>
  <node id="ThermalNoiseRemoval">
    <operator>ThermalNoiseRemoval</operator>
    <sources>
      <sourceProduct refid="Apply-Orbit-File"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <selectedPolarisations>VH,VV</selectedPolarisations>
      <removeThermalNoise>true</removeThermalNoise>
      <reIntroduceThermalNoise>false</reIntroduceThermalNoise>
    </parameters>
  </node>
  <node id="Calibration">
    <operator>Calibration</operator>
    <sources>
      <sourceProduct refid="ThermalNoiseRemoval"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <sourceBands/>
      <auxFile>Product Auxiliary File</auxFile>
      <externalAuxFile/>
      <outputImageInComplex>false</outputImageInComplex>
      <outputImageScaleInDb>false</outputImageScaleInDb>
      <createGammaBand>false</createGammaBand>
      <createBetaBand>false</createBetaBand>
      <selectedPolarisations>VH,VV</selectedPolarisations>
      <outputSigmaBand>true</outputSigmaBand>
      <outputGammaBand>false</outputGammaBand>
      <outputBetaBand>false</outputBetaBand>
    </parameters>
  </node>
  <node id="Terrain-Correction">
    <operator>Terrain-Correction</operator>
    <sources>
      <sourceProduct refid="Calibration"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <sourceBands/>
      <demName>Copernicus 30m Global DEM</demName>
      <externalDEMFile/>
      <externalDEMNoDataValue>0.0</externalDEMNoDataValue>
      <externalDEMApplyEGM>true</externalDEMApplyEGM>
      <demResamplingMethod>BILINEAR_INTERPOLATION</demResamplingMethod>
      <imgResamplingMethod>BILINEAR_INTERPOLATION</imgResamplingMethod>
      <pixelSpacingInMeter>10.0</pixelSpacingInMeter>
      <pixelSpacingInDegree>8.983152841195215E-5</pixelSpacingInDegree>
      <mapProjection>AUTO:42001</mapProjection>
      <alignToStandardGrid>false</alignToStandardGrid>
      <standardGridOriginX>0.0</standardGridOriginX>
      <standardGridOriginY>0.0</standardGridOriginY>
      <nodataValueAtSea>true</nodataValueAtSea>
      <saveSelectedSourceBand>true</saveSelectedSourceBand>
      <outputComplex>false</outputComplex>
      <applyRadiometricNormalization>false</applyRadiometricNormalization>
      <incidenceAngleForSigma0>Use projected local incidence angle from DEM</incidenceAngleForSigma0>
      <incidenceAngleForGamma0>Use projected local incidence angle from DEM</incidenceAngleForGamma0>
      <auxFile>Latest Auxiliary File</auxFile>
      <externalAuxFile/>
    </parameters>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="Terrain-Correction"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>${output_file}</file>
      <formatName>GeoTIFF-BigTIFF</formatName>
    </parameters>
  </node>
</graph>
"""
    graph_path = os.path.join(os.getcwd(), "s1_preprocess_graph.xml")
    with open(graph_path, "w") as f:
        f.write(xml_content)
    return graph_path


def ensure_complete_s1_safe(data_dir):
    """Ensures S1 .SAFE folder is 100% extracted (recovers from Ctrl+C partial extractions)."""
    s1_zips = glob.glob(os.path.join(data_dir, "S1*.zip"))
    safe_folders = glob.glob(os.path.join(data_dir, "*.SAFE"))

    if s1_zips and safe_folders:
        # Verify measurement TIFF sizes match the zip archive
        with zipfile.ZipFile(s1_zips[0], 'r') as zf:
            for info in zf.infolist():
                if info.filename.endswith('.tiff'):
                    extracted_file = os.path.join(data_dir, info.filename.replace('/', os.sep))
                    if not os.path.exists(extracted_file) or os.path.getsize(extracted_file) != info.file_size:
                        print(f"Detected incomplete .SAFE extraction in {data_dir}. Re-extracting cleanly...")
                        shutil.rmtree(safe_folders[0], ignore_errors=True)
                        safe_folders = []
                        break

    if not safe_folders and s1_zips:
        print(f"Auto-extracting Sentinel-1 archive: {os.path.basename(s1_zips[0])}...")
        with zipfile.ZipFile(s1_zips[0], 'r') as zf:
            zf.extractall(data_dir)
        safe_folders = glob.glob(os.path.join(data_dir, "*.SAFE"))

    return safe_folders[0] if safe_folders else None


def process_s1_dir(data_dir, snap_bin):
    print(f"\n==========================================")
    print(f"Running Sentinel-1 preprocessing for: {data_dir}")
    print(f"==========================================")

    output_tif = os.path.join(data_dir, "S1A_Terrain_Corrected.tif")
    if os.path.exists(output_tif) and os.path.getsize(output_tif) > 100_000_000:
        print(f"S1A_Terrain_Corrected.tif already exists in {data_dir}. Skipping.")
        return
    elif os.path.exists(output_tif):
        os.remove(output_tif)

    input_safe = ensure_complete_s1_safe(data_dir)
    if not input_safe:
        print(f"Error: Could not find any Sentinel-1 .SAFE or S1*.zip inside {data_dir}")
        return

    print(f"Found input SAFE folder: {input_safe}")

    graph_xml = create_snap_graph()
    print("Executing SNAP gpt (Graph Processing Tool)...")

    cmd = [
        snap_bin,
        graph_xml,
        f"-Pinput_file={input_safe}",
        f"-Poutput_file={output_tif}"
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Success! Terrain Corrected Sentinel-1 image saved to: {output_tif}")
    finally:
        if os.path.exists(graph_xml):
            os.remove(graph_xml)


def main():
    snap_bin = r"C:\Program Files\snap\bin\gpt.exe"
    if not os.path.exists(snap_bin):
        print(f"Error: ESA SNAP not found at {snap_bin}.")
        sys.exit(1)

    base_dir = r"C:\Users\rnine\LIS"
    target = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(base_dir, "data", "patch1")

    sub_patches = sorted(
        [d.path for d in os.scandir(target) if d.is_dir() and d.name.startswith("patch")],
        key=lambda p: int(p.split("patch")[-1]) if p.split("patch")[-1].isdigit() else 999
    )
    if sub_patches:
        for sp in sub_patches:
            process_s1_dir(sp, snap_bin)
    else:
        process_s1_dir(target, snap_bin)


if __name__ == "__main__":
    main()
