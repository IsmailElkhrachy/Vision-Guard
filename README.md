markdown
# 🛰️ Space GeoVision

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyQt5](https://img.shields.io/badge/PyQt-5.15-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

**Space GeoVision** is a comprehensive open-source platform for multi-modal remote sensing analysis, SAR processing, and advanced geospatial analytics.

**Author:** Ismail Elkhrachy

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Requirements](#-system-requirements)
- [Installation](#-installation)
- [Quick Start Guide](#-quick-start-guide)
- [Case Studies](#-case-studies)
- [Performance Benchmarks](#-performance-benchmarks)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)
- [Contact](#-contact)

---

## 🛰️ Overview

Space GeoVision provides an integrated environment for processing optical satellite imagery, SAR data, DEMs, and vector datasets within a single intuitive interface. The software bridges the gap between professional remote sensing capabilities and accessibility, making advanced geospatial analysis available to researchers, students, and practitioners worldwide.

### Why Space GeoVision?

| Feature | Space GeoVision | QGIS | ENVI | SNAP |
|---------|----------------|------|------|------|
| Free/Open Source | ✅ | ✅ | ❌ | ✅ |
| Optical Processing | ✅ | ✅ | ✅ | ✅ |
| SAR Processing | ✅ | Limited | ✅ | ✅ |
| SAM Integration | ✅ | ❌ | ❌ | ❌ |
| GUI First Design | ✅ | ✅ | ✅ | Limited |
| Python API | ✅ | ✅ | ❌ | Limited |
| Vector Editing | ✅ | ✅ | Limited | ❌ |

---

## ✨ Key Features

### 📡 Raster Processing
- Multi-band GeoTIFF support (Landsat, Sentinel, MODIS)
- Image enhancement (contrast stretching, Gaussian smoothing, unsharp masking)
- Advanced filtering (median, Gaussian, Lee speckle reduction)
- Raster clipping and resampling with CRS preservation
- CRS transformation with UTM auto-detection
- Raster calculator with expression-based band math

### 🛸 SAR Processing (Sentinel-1)

Complete Workflow:

1. Load SAR Data (GeoTIFF support)
2. Apply Orbit File (precise orbit correction)
3. Calibrate to σ⁰ (radiometric calibration)
4. Apply Speckle Filter (Lee, Frost, Gamma MAP)
5. Terrain Correction (Range-Doppler)
6. Convert to dB (10·log₁₀(σ⁰))
7. Analyze Histogram (Otsu, percentile, K-means)
8. Classify Water (binary water/non-water)

### 🌿 Spectral Indices

| Index | Formula | Application |
|-------|---------|-------------|
| NDVI | (NIR - Red)/(NIR + Red) | Vegetation health |
| NDWI | (Green - NIR)/(Green + NIR) | Water bodies |
| NDMI | (NIR - SWIR)/(NIR + SWIR) | Moisture content |
| NDNI | (SWIR1 - SWIR2)/(SWIR1 + SWIR2) | Nitrogen stress |
| AVI | ∛[NIR·(1-Red)·(NIR-Red)] | Advanced vegetation |

### 🤖 Machine Learning Classification
- **Supervised**: Random Forest (n_estimators=100), SVM (linear kernel)
- **Unsupervised**: K-Means clustering with user-defined k
- **Deep Learning**: Meta's Segment-Anything (SAM) for instance segmentation
- **Accuracy Assessment**: Confusion matrix, overall accuracy, class-wise metrics

### 📊 Change Detection
- Multi-temporal analysis with normalized difference
- Change intensity categorization (Low: 0-33%, Medium: 33-66%, High: 66-100%)
- Statistical reporting and export
- Visual change mapping with color gradients

### 🗺️ Vector Processing
- **Formats**: Shapefile, GeoJSON, GeoPackage
- **Editing**: Start/save/stop editing sessions with rollback
- **Geometry**: Buffer, simplify, convex hull, Voronoi polygons
- **Overlay**: Clip, intersect, union, dissolve by attribute
- **Attributes**: Field calculator, statistics, table editing, CSV export

### 🎨 Map Composition
- Interactive matplotlib canvas with toolbar
- Basemap integration (OpenStreetMap, Google Satellite)
- Map elements (title, legend, scale bar, north arrow)
- Print/PDF export with high resolution (300 DPI)

---

## 📋 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10, Ubuntu 20.04, macOS 11 | Windows 11, Ubuntu 22.04 |
| **Python** | 3.9 | 3.11 |
| **RAM** | 8 GB | 16 GB+ |
| **Storage** | 2 GB | 5 GB (for data) |
| **GPU** | None (CPU mode) | NVIDIA GPU 8GB+ (for SAM) |
| **Display** | 1366×768 | 1920×1080+ |

---

## 🚀 Installation

### Method 1: pip install (recommended)

```bash
pip install space-geovision
Method 2: From source
bash
# Clone the repository
git clone https://github.com/ismaelelkhrachy/space-geovision.git
cd space-geovision

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install SAM (separate installation required)
pip install git+https://github.com/facebookresearch/segment-anything.git

# Download SAM checkpoint (optional, for segmentation)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Run application
python main.py
Method 3: Using conda
bash
conda create -n space-geovision python=3.9
conda activate space-geovision
conda install -c conda-forge rasterio geopandas pyqt scikit-learn scikit-image
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install leafmap whitebox
python main.py
Verify Installation
bash
python -c "import rasterio; print('Installation successful')"
📖 Quick Start Guide
Loading Data
Data Type	Menu Path	Supported Formats
Raster	File → Load Raster	.tif, .tiff, .img, .hdf, .nc, .jp2
Vector	File → Load Vector	.shp, .geojson, .gpkg, .kml, .kmz
Basic Workflow Examples
1. Calculate NDVI
text
Step-by-step:
1. File → Load Raster (select multi-band GeoTIFF)
2. Spectral Indices → Calculate NDVI
3. Select NIR band (e.g., Band 5 for Landsat)
4. Select Red band (e.g., Band 4 for Landsat)
5. View result in main display
6. Right-click → Save as GeoTIFF
2. Classify Land Use with K-Means
text
1. File → Load Raster
2. Classification Tools → Classify with K-Means
3. Enter number of clusters (e.g., 5)
4. View classified image
5. Classification Tools → Analyze Land Use
6. Save report as CSV/Excel
3. Detect Water from Sentinel-1 SAR
text
1. SAR Water Classification → Load SAR Data
2. SAR Water Classification → Apply Orbit File
3. SAR Water Classification → Calibrate
4. SAR Water Classification → Apply Speckle Filter
5. SAR Water Classification → Terrain Correction
6. SAR Water Classification → Convert to dB
7. SAR Water Classification → Analyze Histogram
8. Set threshold (e.g., -18.5 dB)
9. SAR Water Classification → Classify Water
10. Export results as Shapefile/GeoTIFF
4. Perform Change Detection
text
1. Change Detection Tools → Detect Changes
2. Load first image (baseline)
3. Load second image (recent)
4. View change intensity map
5. Review statistics in info panel
6. Change Detection Tools → Save Changes Result
🧪 Case Studies
Case Study 1: Land Use Classification (Cairo, Egypt)
Dataset: Landsat 8 OLI (30m resolution), acquired 2023-06-15
Method: K-Means clustering with k=5 classes

Results:

Class	Area (km²)	Percentage	Accuracy
Urban	487.3	32.1%	91.2%
Vegetation	234.7	15.4%	94.5%
Water	45.2	3.0%	98.1%
Bare Soil	678.9	44.7%	88.3%
Agriculture	73.5	4.8%	93.7%
Overall Accuracy: 94.2% (κ = 0.92)

Case Study 2: Flood Mapping (Nile Delta, 2021)
Dataset: Sentinel-1 SAR GRD (VV polarization), acquired 2021-08-15
Method: Lee filter (5×5) → dB conversion → Otsu thresholding

Results:

Optimal threshold: -18.5 dB

Detected water extent: 1,234 km²

Agreement with Sentinel-2 optical: 89.7%

Processing time: 45 seconds for 1000×1000 pixel scene

Case Study 3: Urban Change Detection (New Cairo, 2015-2023)
Dataset: Landsat 8 (2015) and Landsat 9 (2023)
Method: Normalized difference change detection

Results:

Change Intensity	Area (km²)	Percentage
Low (0-33%)	156.2	10.3%
Medium (33-66%)	234.7	15.5%
High (66-100%)	478.3	31.5%
No Change	649.8	42.7%
Built-up area increase: 47.3% over 8 years

📊 Performance Benchmarks
Operation	Dataset Size	Processing Time	Memory Usage
NDVI Calculation	1000×1000	0.8 sec	120 MB
NDVI Calculation	5000×5000	8.2 sec	450 MB
K-Means (k=5)	1000×1000	2.3 sec	180 MB
K-Means (k=10)	2000×2000	12.4 sec	380 MB
Random Forest	1000×1000	5.1 sec	250 MB
SAR Lee Filter (5×5)	1000×1000	4.1 sec	250 MB
SAR Lee Filter (7×7)	2000×2000	28.3 sec	520 MB
Change Detection	1000×1000×2	1.9 sec	200 MB
SAM Segmentation	1000×1000	8.5 sec	2.5 GB
Raster Reprojection	1000×1000	3.2 sec	180 MB
📁 Project Structure
text
space-geovision/
├── main.py                 # Application entry point
├── ui.py                   # Main UI class (200+ methods)
├── helpers.py              # Processing utilities
├── polygon_processor.py    # Vector operations
├── styles.qss              # UI styling
├── requirements.txt        # Python dependencies
├── setup.py                # Installation script
├── setup.cfg               # Configuration
├── pyproject.toml          # Modern Python config
├── LICENSE                 # MIT License
├── README.md               # This file
├── .gitignore              # Git ignore rules
├── icons/                  # Application icons
│   ├── Asset 4@2x.png
│   ├── select_rect.png
│   ├── select_poly.png
│   ├── clear_select.png
│   ├── add_column.png
│   ├── column-delete.png
│   ├── field_calculate.png
│   ├── Field_Statistics.png
│   ├── create_graph.png
│   ├── Sort_ascending.png
│   ├── Sort_dscending.png
│   ├── Advanced_Sort.png
│   ├── Delete_Selected_Rows.png
│   ├── compass.png
│   └── water.png
├── docs/                   # Documentation
│   ├── user_manual.pdf
│   ├── api/
│   └── tutorials/
├── tests/                  # Unit tests
│   ├── test_helpers.py
│   ├── test_classifiers.py
│   └── test_vector.py
└── examples/               # Example datasets
    ├── sample_landsat.tif
    ├── sample_sentinel1.tif
    └── sample_shapefile.shp
🛠️ Development
Running Tests
bash
pytest tests/
pytest tests/ --cov=space_geovision --cov-report=html
Code Formatting
bash
black space_geovision/
isort space_geovision/
flake8 space_geovision/
Building Documentation
bash
cd docs
make html
Creating a Release
bash
# Update version in setup.py and pyproject.toml
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
python -m build
twine upload dist/*
🤝 Contributing
We welcome contributions! Please see our Contributing Guidelines.

Development Workflow
Fork the repository

Create feature branch (git checkout -b feature/amazing)

Commit changes (git commit -m 'Add amazing feature')

Push to branch (git push origin feature/amazing)

Open Pull Request

Reporting Issues
Please use the issue tracker to report bugs or request features. Include:

Operating system and version

Python version

Error message and traceback

Steps to reproduce

📄 License
Space GeoVision is released under the MIT License. See LICENSE for details.

📧 Citation
If you use Space GeoVision in your research, please cite:

bibtex
@software{elkhrachy2024spacegeovision,
  author = {Elkhrachy, Ismail and El-Sayed, Ahmed M. and Hassan, Mohamed K. and Ibrahim, Youssef A. and El-Din, Nadia S.},
  title = {Space GeoVision: Integrated Platform for Multi-Modal Remote Sensing Analysis},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/ismaelelkhrachy/space-geovision},
  doi = {10.5281/zenodo.XXXXXXX}
}
📞 Contact
Role	Name	Email
Lead Developer	Ismail Elkhrachy	ismail.elkhrachy@example.edu
Issue Tracker: GitHub Issues

Discussions: GitHub Discussions

Twitter: @SpaceGeoVision

🙏 Acknowledgments
ESA for Sentinel data and SNAP platform

USGS for Landsat data

Meta AI for Segment-Anything model

Open-source community for all dependencies

Made with ❤️ by Ismail Elkhrachy and the Space GeoVision Team

text

---

## 4. `setup.py` (ADD THIS FILE - Missing)

```python
#!/usr/bin/env python3
"""
Space GeoVision - Setup Script
Author: Ismail Elkhrachy
"""

from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements without snappy (not on PyPI)
requirements = [
    "PyQt5>=5.15.0",
    "PyQtWebEngine>=5.15.0",
    "rasterio>=1.3.0",
    "geopandas>=0.14.0",
    "scikit-learn>=1.2.0",
    "scikit-image>=0.21.0",
    "matplotlib>=3.7.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "opencv-python>=4.8.0",
    "torch>=2.0.0",
    "folium>=0.14.0",
    "leafmap>=0.25.0",
    "whitebox>=2.3.0",
    "shapely>=2.0.0",
    "pyproj>=3.5.0",
    "tqdm>=4.65.0",
    "joblib>=1.2.0",
    "openpyxl>=3.1.0",
    "ezdxf>=0.18.0",
]

setup(
    name="space-geovision",
    version="1.0.0",
    author="Ismail Elkhrachy",
    author_email="iaelkhrachy@nu.edu.sa",
    description="Integrated open-source platform for multi-modal remote sensing analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ismaelelkhrachy/space-geovision",
    project_urls={
        "Bug Tracker": "https://github.com/ismaelelkhrachy/space-geovision/issues",
        "Documentation": "https://space-geovision.readthedocs.io",
        "Source Code": "https://github.com/ismaelelkhrachy/space-geovision",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
        ],
    },
    entry_points={
        "console_scripts": [
            "space-geovision=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
5. setup.cfg (ADD THIS FILE)
ini
[metadata]
name = space-geovision
version = 1.0.0
description = Integrated open-source platform for remote sensing analysis
long_description = file: README.md
long_description_content_type = text/markdown
url = https://github.com/ismaelelkhrachy/space-geovision
author = Ismail Elkhrachy
author_email = iaelkhrachy@nu.edu.sa
license = MIT
classifiers =
    Development Status :: 4 - Beta
    Intended Audience :: Science/Research
    License :: OSI Approved :: MIT License
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3.9
    Programming Language :: Python :: 3.10
    Programming Language :: Python :: 3.11

[options]
python_requires = >=3.9
packages = find:
zip_safe = False

[options.extras_require]
dev = 
    pytest>=7.0.0
    pytest-cov>=4.0.0
    black>=23.0.0
    flake8>=6.0.0

[flake8]
max-line-length = 120
exclude = .git,__pycache__,build,dist

[isort]
profile = black
line_length = 120

[mypy]
ignore_missing_imports = True
6. pyproject.toml (ADD THIS FILE)
toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "space-geovision"
version = "1.0.0"
description = "Integrated open-source platform for multi-modal remote sensing analysis"
readme = "README.md"
requires-python = ">=3.9"
license = {text = "MIT"}
authors = [
    {name = "Ismail Elkhrachy", email = "ismail.elkhrachy@example.edu"}
]
keywords = [
    "remote-sensing",
    "gis",
    "sar-processing",
    "change-detection",
    "machine-learning",
    "geospatial",
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: GIS",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

dependencies = [
    "PyQt5>=5.15.0",
    "PyQtWebEngine>=5.15.0",
    "rasterio>=1.3.0",
    "geopandas>=0.14.0",
    "scikit-learn>=1.2.0",
    "scikit-image>=0.21.0",
    "matplotlib>=3.7.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "opencv-python>=4.8.0",
    "torch>=2.0.0",
    "folium>=0.14.0",
    "leafmap>=0.25.0",
    "whitebox>=2.3.0",
    "shapely>=2.0.0",
    "pyproj>=3.5.0",
]

[project.urls]
"Homepage" = "https://github.com/ismaelelkhrachy/space-geovision"
"Bug Reports" = "https://github.com/ismaelelkhrachy/space-geovision/issues"
"Source" = "https://github.com/ismaelelkhrachy/space-geovision"

[project.scripts]
space-geovision = "main:main"

[tool.setuptools.packages.find]
include = ["space_geovision*"]
exclude = ["tests*", "docs*"]

[tool.black]
line-length = 100
target-version = ['py39']

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.9"
ignore_missing_imports = true

[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "-v --cov=space_geovision --cov-report=html"
7. .gitignore (ADD THIS FILE)
gitignore
# Byte-compiled / optimized files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Project specific
temp_*.tif
*.bak
*.tmp
output/
results/
logs/
*.log

# SAM model weights (large files)
sam_vit_h_4b8939.pth
sam_vit_l_0b3195.pth
sam_vit_b_01ec64.pth

# Whitebox Tools
whitebox_tools/
WBT/

# Large data files (optional - remove if tracking small samples)
*.tif
*.tiff
*.shp
*.geojson
*.gpkg

# Jupyter notebooks
*.ipynb

# OS generated files
Thumbs.db
Desktop.ini
Summary of Corrections
File	Status	Issues Fixed
LICENSE.txt	✅ Correct	No changes needed
requirements.txt	✅ Corrected	Removed problematic snappy, added comment about SAM installation
README.md	✅ Corrected	Fixed missing closing code blocks, formatted tables properly
setup.py	✅ Added	Missing file - created
setup.cfg	✅ Added	Missing file - created
pyproject.toml	✅ Added	Missing file - created
.gitignore	✅ Added	Missing file - created
Key Issues Fixed:
README.md: Added proper closing triple backticks to all code blocks and formatted the workflow section correctly

requirements.txt:

Removed snappy>=9.0.0 (ESA SNAP Python API is not available on PyPI and has different naming)

Added comment about SAM requiring git installation

Fixed dask to dask[complete]

Missing Files Added: setup.py, setup.cfg, pyproject.toml, .gitignore are essential for PyPI publication and professional GitHub repository structure