#set page(paper: "a4", margin: 1in)
#set text(font: "Linux Libertine", size: 11pt)
#set heading(numbering: "1.1")
#set par(justify: true)

#align(center)[
  #text(size: 16pt, weight: "bold")[Thesis Summary & Experimental Pipeline] \
  #vspace(1em)
  #text(size: 12pt)[Weakly Supervised Snow Cover Segmentation in the Himalayas]
]

#vspace(1em)

= Data Engineering & Preprocessing
The dataset is constructed from Sentinel-1 (C-Band SAR) inputs paired with Sentinel-2 (Optical) weak labels generated via the Let-It-Snow (LIS) algorithm. The study area is the Himalayas, requiring rigorous handling of clouds and topographical distortions.

- *Data Volume & Split:* 12 total satellite passes. Patches 1-6 are snow-free Summer references (~31 GB). Patches 7-12 are snow-covered Winter images (~18.5 GB) utilized for active training. Total raw data footprint is ~50 GB.
- *Pipeline Optimization:* Standard patch extraction would yield >100 GB of arrays. Instead, the pipeline employs *Internally Tiled GeoTIFFs* for on-the-fly $256 times 256$ window loading directly from the 18.5 GB compressed binaries. 
- *Label Noise & Filtration:* The labels are heavily clouded (up to 92% background). Clouded pixels are dynamically mapped to a PyTorch `ignore_index` (255) to train selectively on visible data. Spatial voids (28% of the bounding box) are systematically filtered out, yielding an optimized dataset of ~26,650 valid windows per winter pass.

= Phase 1: C-Band Baseline & Stratified Evaluation
*Objective:* Establish the baseline performance of C-Band SAR for binary snow cover classification. \
*Evaluation:* Because radar scattering varies drastically with altitude, performance (Snow IoU and F1-Score) is evaluated strictly across elevation strata (<3000m, 3000-4000m, 4000-5000m, >5000m) using an aligned 10m Copernicus DEM.

*Ablation Experiments:*
1. *Baseline U-Net:* Standard architecture utilizing only backscatter (VV + VH).
2. *U-Net + MLE:* Incorporating Maximum Likelihood Estimation (MLE) priors.
3. *U-Net + MLE + DEM:* Early fusion of topographical data (Elevation, Slope) as input channels.
4. *Attention U-Net:* Utilizing attention gates to force the network to focus on structural terrain features rather than radar speckle.

= Phase 2: C-Band vs. S-Band Discrepancy
*Objective:* Prove the penetration superiority of S-Band SAR in specific topologies. \
*Rationale:* C-Band wavelengths fail to penetrate dense Himalayan forest canopies due to volumetric scattering. This phase trains identical architectures on S-Band data to benchmark performance deltas against Phase 1, explicitly isolating the elevation and vegetation-density bins where S-Band mathematically outperforms C-Band.

= Phase 3: Dynamic Multi-Band Fusion & Temporal Analysis (Novelty)
*Objective:* Architect a novel multi-modal network capable of dynamically combining S-Band and C-Band data, and deploy it for temporal mapping.

*Novel Modeling Aspects:*
- *Dynamic Fusion:* Rather than ordinary early-fusion stacking, we propose a *Dual-Encoder U-Net* (Late Fusion) or a *Squeeze-and-Excitation* weighting mechanism. The network will learn to dynamically route spatial attention to the S-Band in dense forests, and the C-Band in barren high-altitude zones.
- *Temporal Generalization:* The finalized fusion model will be deployed on a continuous time-series (subsequent year) to map temporal snowmelt dynamics, proving the model's robustness to spatio-temporal shifts.
