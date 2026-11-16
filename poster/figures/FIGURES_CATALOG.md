# LIS Project Figures & Diagrams Catalog

This catalog indexes all images, diagrams, flowcharts, and plots gathered across the project repository into `poster/figures/` for your analysis and inclusion in the Inria Poster.

Both vector formats (`.pdf`), high-resolution raster formats (`.png`), and source diagrams (`.d2`) are available.

---

## 1. Flowcharts & System Architecture Diagrams (Top Candidates for Methodology Section)

| Figure / File Name | Formats | Description & Role in Poster | Recommended Poster Location |
| :--- | :--- | :--- | :--- |
| **`pipeline_diagram`** | `.pdf`, `.png`, `.d2` | **End-to-End System Flowchart**: Shows raw Sentinel-1 SAR + Sentinel-2 MSI + Copernicus DEM ingestion, LIS weak label generation, U-Net segmentation, and all-weather cloud-penetrating snow output. | **Methodology / Workflow** (Center/Top column) |
| **`phase2_fusion_diagram`** | `.pdf`, `.png`, `.d2` | **Deep Learning Architecture Diagram**: Illustrates the multi-modal fusion architecture combining SAR C-band channels, Rayleigh summer reference, and DEM topography. | **Model Architecture** |
| **`data_preprocessing_diagram`** | `.pdf`, `.png`, `.d2` | **Data Pipeline Flowchart**: Details radiometric calibration, terrain correction, thermal noise removal, and cloud/permasnow masking. | **Data Preparation / Pipeline** |
| **`weak_supervision_concept`** | `.pdf`, `.png`, `.d2` | **Weak Supervision Conceptual Diagram**: Shows how optical LIS masks provide labels for training cloud-penetrating SAR models. | **Introduction / Concept** |

---

## 2. Qualitative Results & Visual Demonstrations (High Visual Impact for Poster)

| Figure / File Name | Formats | Description & Visual Impact | Recommended Poster Location |
| :--- | :--- | :--- | :--- |
| **`model_outputs_comparison`** | `.pdf`, `.png` | **Multi-Model Benchmark Output**: Compares ground truth (with and without cloud obscuration) against Model 1, 2, and 3 predictions. Clearly proves SAR recovers snowpack obscured by clouds. | **Core Results (Hero Figure)** |
| **`sentinel2_lis_pipeline_sample`** | `.pdf`, `.png` | **Optical Snow & Cloud Pipeline**: Shows 4 subplots: (a) RGB True Color, (b) SWIR False Color, (c) NDSI Index, and (d) LIS Snow/Cloud mask. | **Data / Optical Baseline** |
| **`sample_satellite_data`** | `.pdf`, `.png` | **Multi-Modal Data Inputs**: Shows SAR backscatter ($\sigma^0_{\text{VV}}$, $\sigma^0_{\text{VH}}$), summer Rayleigh reference, DEM elevation & slope, and optical weak labels. | **Dataset & Inputs Column** |
| **`multitemporal_winter_series`** | `.pdf`, `.png` | **Multi-Temporal Winter Evolution**: Time-series comparison from Dec to Feb showing Sentinel-1 SAR penetrating thick clouds on Jan 26 while Sentinel-2 is blinded. | **Temporal Robustness / Findings** |
| **`lis_without_cloud_mask_comparison`** | `.pdf`, `.png` | **Cloud Masking Impact Analysis**: Compares raw LIS classification with and without cloud masks over complex high-altitude terrain. | **Ablation / Discussion** |
| **`patch12_second_test_image_lis_cloud_analysis`** | `.pdf`, `.png` | **Cloud-Snow Ambiguity Case Study**: High-resolution diagnostic of patch 12 under challenging snow/cloud conditions. | **Detailed Diagnostics** |

---

## 3. Quantitative Plots & Experimental Evaluations (For Results & Discussion Blocks)

| Figure / File Name | Formats | Description | Recommended Poster Location |
| :--- | :--- | :--- | :--- |
| **`precision_recall_comparison_plot`** | `.pdf`, `.png` | Precision, Recall, F1-Snow, and Macro F1 across Models 1, 2, and 3 on validation and spatial transfer domains. | **Quantitative Results / Tables** |
| **`elevation_performance_plot`** | `.pdf`, `.png` | Stratified snow performance across elevation bands ($<4000\text{m}$ to $>6000\text{m}$) demonstrating high-altitude reliability. | **Topographic Analysis** |
| **`domain_shift_plot`** | `.pdf`, `.png` | Snow elevation distributions revealing an $871\text{m}$ snowline elevation shift between training and unseen test regions. | **Transferability / Domain Shift** |
| **`scene_cloud_snow_plot`** | `.pdf`, `.png` | Scene-wise valid snow percentage versus cloud obscuration percentages. | **Dataset Statistics** |
| **`sar_backscatter_distribution_plot`** | `.pdf`, `.png` | Empirical PDF of winter-summer backscatter contrast ($\Delta\sigma^0$) for snow vs no-snow vs Nagler threshold. | **Radar Physics / Theory** |
| **`threshold_sensitivity_plot`** | `.pdf`, `.png` | Diagnostic sensitivity curve for decision threshold $\tau$ under spatial shift. | **Model Robustness** |
| **`analysis_plots`** | `.pdf`, `.png` | Comprehensive multi-panel performance analysis summary. | **Summary Results** |

---

## 4. Upstream LIS & Satellite Reference Figures

- **`reference_lis/`**: 23 figures from CESBIO's official Let-It-Snow ATBD and documentation (e.g., S2 snow algorithms, interpolation comparisons, MODIS/S2 annual maps).
- **`satellite_quicklooks/`**: Sample quicklook previews from raw Sentinel-1 GRD and Sentinel-2 MSI products in the dataset.

---

## How to Include in `inria-poster.tex`

To insert any figure into your poster:
```latex
\begin{plainblock}{Pipeline Architecture}
  \centering
  \includegraphics[width=\linewidth]{figures/pipeline_diagram.pdf}
\end{plainblock}
```
*(Vector `.pdf` is recommended for maximum print sharpness on large A0 format; `.png` is also fully supported).*
