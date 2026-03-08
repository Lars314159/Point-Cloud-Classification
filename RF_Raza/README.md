# Tinto Block Cross-Validation

Spatial block cross-validation for hyperspectral mineral classification using Random Forest on the [Tinto drill core dataset](https://rodare.hzdr.de/record/2256).

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
tinto-block-cv/
├── TintoBlockCV.py    # Main class: data loading, blocking, cross-validation
├── model_rf.py        # Random Forest factory function
├── main.ipynb         # Jupyter notebook walkthrough
├── requirements.txt   # Python dependencies
└── README.md
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Tinto dataset

Follow the instructions at [hylite documentation](https://rodare.hzdr.de/record/2256) to obtain the `tinto3D.hyc` file. Place it in a `data/` directory (or update the path in the notebook).

### 3. Run the demo

```bash
jupyter notebook demo.ipynb
```

## Quick Start

```python
from TintoBlockCV import TintoBlockCV

tbcv = TintoBlockCV()

# Load data
tbcv.load_dataset(
    path='data/tinto3D.hyc',
    subset='synth',
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
