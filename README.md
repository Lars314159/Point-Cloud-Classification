<<<<<<< HEAD
<<<<<<< HEAD
# Tinto Block Cross-Validation

Spatial block cross-validation for hyperspectral mineral classification using Random Forest on the [Tinto drill core dataset]([https://hylite.readthedocs.io/](https://rodare.hzdr.de/record/2256)).

## Why Block Cross-Validation?

Standard k-fold cross-validation randomly splits samples, which leaks spatial information — nearby points end up in both train and test sets, inflating accuracy. Block cross-validation splits the drill core into **spatially contiguous blocks** so that the test set is always a region the model has never seen. This gives a more realistic estimate of how well the model generalizes to new areas of the core.

## Features

- **Multiple blocking strategies**: KMeans spatial clustering, horizontal (z-axis) slicing, or vertical (x/y-axis) slicing
- **Configurable Random Forest**: full control over tree count, depth, leaf size, class weighting, and OOB scoring
- **Preprocessing options**: z-score normalization, PCA dimensionality reduction, spectral derivative features
- **Interactive 3D visualization**: view the point cloud colored by class labels, spatial blocks, or prediction accuracy
- **Automated experiment logging**: results saved to CSV with all hyperparameters for reproducibility

## Project Structure

```
Point-cloud-classification/
├── TintoBlockCV.py    # Main class: data loading, blocking, cross-validation
├── model_rf.py        # Random Forest factory function
├── main.ipynb         # Jupyter notebook walkthrough
├── requirements.txt   # Python dependencies
├── LICENSE            # MIT License
└── README.md
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Tinto dataset

Follow the instructions at [hylite documentation]([https://hylite.readthedocs.io/](https://rodare.hzdr.de/record/2256)) to obtain the `tinto3D.hyc` file. Place it in a `data/` directory (or update the path in the notebook).

### 3. Run the main

```bash
jupyter notebook main.ipynb
```

## Quick Start

```python
from TintoBlockCV import TintoBlockCV

tbcv = TintoBlockCV()

# Load data
tbcv.load_dataset(
    path='data/tinto3D.hyc',
    subset='real',
    wave_lengths='LWIR',
    n_samples=50000,
)

# Configure model
tbcv.set_model_parameters(
    n_estimators=300,
    max_depth=30,
    min_samples_leaf=5,
    class_weight='balanced',
    oob_score=True,
)

# Run cross-validation across block sizes
tbcv.crossvalidate(
    n_blocks_list=[50, 20, 10, 6, 5],
    blocking_method='kmeans',
    output_path='results.csv',
)

# Plot results
results = TintoBlockCV.load_from_csv('results.csv')
TintoBlockCV.plot_data(results, y_axis='acc_test')
```

## Blocking Methods

| Method | Description | Best for |
|---|---|---|
| `kmeans` | Clusters points in 3D space using KMeans | General use, irregularly shaped cores |
| `horizontal` | Slices along the z-axis | Simulating depth-based prediction |
| `vertical_x` | Slices along the x-axis | Testing lateral generalization |
| `vertical_y` | Slices along the y-axis | Testing lateral generalization |

## License

MIT
=======
# 🌍 Tinto Hyperspectral Point Cloud Segmentation

This project implements a robust benchmarking pipeline for geological classification on 3D point clouds. It addresses the issue of **spatial autocorrelation** in geospatial data by using a **Block-wise Exclusion Strategy** instead of standard random cross-validation.

## 📌 Project Overview
- **Dataset:** Tinto (Multisensor Benchmark for 3-D Hyperspectral Point Cloud Segmentation).
- **Goal:** Classify rock types (lithology) based on Tinto hyperspectral signatures.
- **Problem:** Standard random splits leak information in spatial data.
- **Solution:** Implemented spatial block-wise cross-validation (holding out 2% to 20% distinct spatial blocks) to test the model's ability to generalize to new, unseen mining areas.
- **Paper:** [DOI: 10.1109/TGRS.2023.3340293](https://doi.org/10.1109/TGRS.2023.3340293)

## 🔧 Technologies
- **Language:** Python 3.11
- **Libraries:** Scikit-learn, Pandas, NumPy, Plyfile
=======
# Point-Cloud-Classification

as part of  
**Data Analysis Project**

by  
*Lars Wunderlich*  
*Hussnain Raza*

# Task
implement at least 2 classifyiers (1 from the paper) but using
block-wise exclusion strategies, with different block sizes (from 2% to
20% of the image within a block)

# Paper
https://doi.org/10.1109/TGRS.2023.3340293
>>>>>>> 12d88423c301116ece5246dd6e119e4abcaef93f
