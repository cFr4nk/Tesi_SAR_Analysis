# Tesi_SAR_Analysis
 
Time-series analysis of Sentinel-1 SAR data over agricultural parcels in **Flevoland (The Netherlands)**, combined with ground-campaign measurements, Sentinel-2 NDVI and rainfall records.
 
The goal is to relate SAR observables — backscatter (σ⁰, VV/VH) and 6-day interferometric coherence — to crop and soil variables (plant height, phenological stage, soil moisture), and to test a change-detection retrieval of Surface Soil Moisture (SSM).
 
> Master's thesis work — Universitat de València.
 
---
 
## Overview
 
The workflow is organised as a chain of Jupyter notebooks:
 
| Notebook | Purpose |
|---|---|
| `Areas_Parcs.ipynb` | Reads the field shapefiles, reprojects them to the Dutch national grid (EPSG:28992) and computes the area of each parcel. Produces `areas_Flavoland.csv` (parcel code, name, crop type, area). |
| `Read_sigma.ipynb` | Extracts **backscatter** statistics. Builds a raster mask per parcel from the shapefiles, applies binary erosion to remove edge effects, reads only the relevant subregion of each ENVI image, and stores mean/std per date and polarisation (VV, VH). Output: `TimeSeries_sigma.csv`. |
| `Read_Cohe.ipynb` | Same procedure applied to the **6-day interferometric coherence** stack. Output: `TimeSeries_Coherencia.csv`. Also loads the rainfall series and resamples it on the 6-day SAR revisit. |
| `NDVI.ipynb` | Retrieves Sentinel-2 L2A (B04, B08) through the **openEO / Copernicus Data Space** API, computes NDVI, aggregates it spatially per parcel (cloud cover ≤ 10%) and exports one CSV per field. |
| `moist_interpol.ipynb` | Interpolates the in-situ soil moisture to a daily series (**PCHIP**), merges it with the backscatter time series, runs per-parcel regressions and applies an **alpha-approximation recursion** to retrieve SSM from consecutive σ⁰ acquisitions. Includes outlier filtering (±3·RMSE) and rainfall cross-checks. |
| `Regressions.ipynb` | Regressions between coherence/backscatter and PCHIP-interpolated **crop height**, grouped by crop type and polarisation (R², Pearson R, p-value, RMSE). |
| `PDFs.ipynb` | Plotting layer: multi-page PDF reports combining SAR time series, rainfall bars, plant height and phenology, both per parcel and aggregated by crop type. |
 
Crop types covered: wheat, beets, potatoes, corn, grassland.
 
---
 
## Data
 
The notebooks expect an external data folder (**not included in this repository**, because of size and licensing):
 
```
App/
├── Data/
│   └── Ground_Campaign/
│       └── Flevoland_data/Data_25_fields/
│           ├── Flevoland-fields-Shapefiles/   # parcel polygons (.shp)
│           ├── Average_Soil_moisture.xlsx     # in-situ SSM
│           └── Phenology_stages.xlsx          # phenological stages
└── Documents/
    └── csv/                                   # intermediate + output CSVs
```
 
SAR stacks (ENVI `.hdr` / `.img`) are read from a separate drive:
 
```
BACKSCATTER/    # calibrated sigma0, VV & VH
COHERENCIA/     # 6-day interferometric coherence
```
 
**Sources**
- Sentinel-1 SLC/GRD — Copernicus (backscatter and coherence pre-processed externally).
- Sentinel-2 L2A — via [openEO](https://openeo.dataspace.copernicus.eu).
- Rainfall — GHCN stations, retrieved through Sensoto.
- Ground campaign — Flevoland 25-fields dataset (2017 season).
---
 
## Requirements
 
Python ≥ 3.10.
 
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn \
            geopandas rasterio pyproj spectral pyshp openpyxl \
            pyarrow openeo deep-translator
```
 
> ⚠️ **Missing module.** All notebooks `import Functions`, a local helper module providing
> `get_input_path()`, `read_img()`, `read_img2()`, `axes_from_metadata()`,
> `open_shp_with_geopandas()`, `read_dates_from_stack()` and `handle_ranges()`.
> This file is not currently tracked in the repository and must be added (or provided
> separately) before the notebooks can run.
 
Paths are currently hard-coded in several cells; adapt `app_path` / `sentinel_path`
to your own environment.
 
---
 
## Suggested execution order
 
```
Areas_Parcs.ipynb          →  parcel geometry & crop table
Read_sigma.ipynb           →  TimeSeries_sigma.csv
Read_Cohe.ipynb            →  TimeSeries_Coherencia.csv
NDVI.ipynb                 →  NDVI/*.csv
moist_interpol.ipynb       →  SSM retrieval + regressions
Regressions.ipynb          →  height vs SAR regressions
PDFs.ipynb                 →  final PDF reports
```
 
---
 
## Method notes
 
- **Parcel masking.** Field polygons are converted to `matplotlib.path.Path` objects and rasterised on the SAR grid. A 3×3 binary erosion is applied so that boundary pixels (mixed / contaminated by neighbouring fields) are excluded from the statistics.
- **Memory handling.** SAR scenes are never fully loaded: only the bounding-box subregion of each parcel is read from disk (`read_subregion`).
- **Temporal alignment.** Sparse in-situ measurements are interpolated to daily values with PCHIP (shape-preserving, no overshoot), then matched to SAR acquisition dates.
- **SSM retrieval.** A recursive alpha-approximation propagates soil moisture from one acquisition to the next using the linear ratio of consecutive backscatter values, re-initialising on an in-situ value when the gap between acquisitions exceeds 7 days.
---
 
## Outputs
 
- `TimeSeries_sigma.csv`, `TimeSeries_Coherencia.csv` — per-parcel SAR statistics (mean, std, date, band).
- `moisture_reg_xfield.csv` — regression metrics per parcel and polarisation.
- PDF reports: `TS_Coherence_Rain.pdf`, `TS_Sigma_Rain_Height.pdf`, `TS_Sigma_Rain_Phenology.pdf`, `Time_Series_Type_Aggregated.pdf`, `Regression_Moisture_xField.pdf`.
---
 
## Author
 
**cFr4nk** — F. Chiapperino
 
