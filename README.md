# GLOF-ID: Automated Mapping and Vulnerability Assessment of Glacial Lakes in Cordillera Blanca, Peru

**Glacial Lake Outburst Floods (GLOFs)** pose a significant threat to high-mountain communities and infrastructure. This project presents an automated method for mapping glacial lakes in the **Cordillera Blanca (Peru)** using satellite imagery and a segmentation-based deep learning model.

## 🛰️ Overview

This repository contains the implementation of our method to:
- Automatically detect and delineate glacial lakes using Sentinel-2 imagery.
- Perform multitemporal analysis (2016–2024) to track changes in lake extent.
- Identify potentially vulnerable lakes based on surface expansion.
- Simulate possible GLOF impacts using lake and topographic parameters.

Our method combines **Sentinel-2 enhanced band composites** with the **Segment Anything Model (SAM v2.1)** to achieve high-accuracy lake extraction, mapping **80% of 448 manually identified glacial lakes**.

## 🌍 Case Study: Cordillera Blanca, Peru

- Period analyzed: **May 2016 vs May 2024**
- Identified 5 lakes with significant surface area expansion:
  - Lake Parón (+13.79 ha)
  - Lake Piticocha (+3.72 ha)
  - 3 additional smaller lakes with notable growth
- GLOF impact simulations show possible risks to:
  - Urban areas (e.g., **Caraz city**)
  - Agricultural zones

## 🧰 Features

- 🔍 Automatic glacial lake segmentation
- 🗓️ Temporal lake change detection
- 🛠️ GLOF vulnerability analysis & visualization

## 🏗️ Repository Structure

```plaintext
GLOF-ID/
│
├── data/                   # Raw and processed datasets (Sentinel-2, DEMs, masks)
│   ├── sentinel/           # Sentinel-2 images (2016, 2024)
│   └── lakes/              # Lake masks and annotations
│
├── notebooks/              # Jupyter notebooks for exploration and visualization
│   └── lake_expansion_analysis.ipynb
│
├── src/                    # Core source code
│   ├── preprocessing/      # Image preprocessing (cloud masking, band composition)
│   ├── segmentation/       # SAM 2.1 lake segmentation
│   ├── postprocessing/     # Filtering and lake polygon extraction
│   └── analysis/           # GLOF risk analysis and visualization
│
├── results/                # Outputs: maps, lake areas, charts
│   └── figures/            # Paper-ready figures
│
├── models/                 # SAM weights and configurations
│
├── requirements.txt        # Dependencies
├── README.md               # Project overview