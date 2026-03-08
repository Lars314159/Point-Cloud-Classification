"""
TintoBlockCV - Spatial Block Cross-Validation for Hyperspectral Rock Classification

This module implements block cross-validation using spatial clustering on the
Tinto hyperspectral drill core dataset. It supports Random Forest classification
with configurable blocking strategies (KMeans, axis-based splits) to evaluate
model generalization across spatially distinct regions.
"""

import os
import csv

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as plot
from hylite import io
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
from tqdm.notebook import tqdm

from model_rf import create_random_forest


class TintoBlockCV:

    # Valid options for dataset loading
    VALID_SUBSETS = ('real', 'synth', 'degr')
    VALID_WAVELENGTHS = ('VNIR', 'SWIR', 'LWIR')
    VALID_LABELS = ('basic', 'complete')

    def __init__(self):
        """Initialize with default model parameters."""
        # Default Random Forest hyperparameters
        self.n_estimators = 100
        self.max_depth = None
        self.min_samples_leaf = 1
        self.max_features = 'sqrt'
        self.class_weight = None
        self.oob_score = False

        # State flags
        self.pca_applied = False
        self.last_oob_score = None
        self.last_feature_importances = None

    def load_dataset(
        self,
        path: str,
        subset: str,
        wave_lengths: str,
        labels: str = 'complete',
        n_samples: int | None = None,
        normalize: bool = True,
    ):
    
        if n_samples is not None and n_samples <= 0:
            raise ValueError("n_samples must be a positive integer or None.")

        if subset not in self.VALID_SUBSETS:
            raise ValueError(f"subset must be one of {self.VALID_SUBSETS}, got '{subset}'.")
        if wave_lengths not in self.VALID_WAVELENGTHS:
            raise ValueError(f"wave_lengths must be one of {self.VALID_WAVELENGTHS}, got '{wave_lengths}'.")
        if labels not in self.VALID_LABELS:
            raise ValueError(f"labels must be one of {self.VALID_LABELS}, got '{labels}'.")

        dataset = io.load(os.path.abspath(path))
        self.class_names = dataset.labels_complete.header.get_list('class names')

        # Select label set
        label_map = {
            'basic': dataset.labels_basic,
            'complete': dataset.labels_complete,
        }
        lab = label_map[labels].data[:, 0]

        # Select data subset
        subset_map = {
            'real': dataset.real,
            'synth': dataset.synth,
            'degr': dataset.degr,
        }
        data = subset_map[subset]

        # Select wavelength range
        wavelength_map = {
            'VNIR': data.vnir,
            'SWIR': data.swir,
            'LWIR': data.lwir,
        }
        data = wavelength_map[wave_lengths]

        features = data.data
        xyz = data.xyz
        rgb = data.rgb
        n_total_samples = features.shape[0]

        # Sample or shuffle indices
        if n_samples is not None:
            if n_samples > n_total_samples:
                raise ValueError(
                    f"n_samples ({n_samples}) exceeds available samples ({n_total_samples})."
                )
            idx = np.random.choice(n_total_samples, size=n_samples, replace=False)
        else:
            idx = np.arange(n_total_samples)
            np.random.shuffle(idx)

        self.features = features[idx, :]
        self.labels = lab[idx].astype(np.int64)
        self.xyz = xyz[idx]
        self.rgb = rgb[idx]
        self.n_samples = self.features.shape[0]
        self.n_features = self.features.shape[1]
        self.n_classes = len(self.class_names)
        self.subset = subset
        self.label_string = labels
        self.wave_lengths = wave_lengths
        self.pca_applied = False

        self._generate_label_rgb()
        if normalize:
            self._normalize_data()

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _generate_label_rgb(self):
        """Assign a random RGB color to each class for visualization."""
        colors = np.random.randint(0, 256, (self.n_classes, 3))
        self.rgb_labels = np.zeros((self.n_samples, 3))
        for i in range(self.n_samples):
            self.rgb_labels[i, :] = colors[self.labels[i], :]

    def _normalize_data(self):
        """Z-score normalize features (zero mean, unit variance per band)."""
        mean = self.features.mean(axis=0)
        std = self.features.std(axis=0)
        std[std == 0] = 1.0  # Avoid division by zero for constant bands
        self.features = (self.features - mean) / std

    def apply_pca(self, n_components: int | None = None, variance_threshold: float = 0.99):
        """
        Reduce dimensionality with PCA on hyperspectral bands.

        :param n_components: Exact number of components (overrides variance_threshold).
        :param variance_threshold: Fraction of variance to retain (default: 0.99).
        """
        target = n_components if n_components is not None else variance_threshold
        pca = PCA(n_components=target, random_state=42)

        self.features = pca.fit_transform(self.features)
        self.n_features = self.features.shape[1]
        self.pca_applied = True
        self.pca_model = pca
        self.pca_variance_explained = pca.explained_variance_ratio_.sum()

        print(
            f"PCA: {pca.n_components_} components retained "
            f"(from {pca.n_features_in_} original bands), "
            f"explaining {self.pca_variance_explained:.2%} of variance"
        )

    def add_spectral_derivatives(self):
        """
        Append first-order spectral derivatives as additional features.

        Derivatives highlight absorption edges better than raw reflectance
        and can improve classification of mineralogically similar classes.
        """
        derivatives = np.diff(self.features, axis=1)
        self.features = np.hstack([self.features, derivatives])
        self.n_features = self.features.shape[1]
        print(
            f"Added {derivatives.shape[1]} derivative features. "
            f"Total features: {self.n_features}"
        )

    # ------------------------------------------------------------------
    # Model configuration
    # ------------------------------------------------------------------

    def set_model_parameters(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        max_features: str | float = 'sqrt',
        class_weight: str | None = None,
        oob_score: bool = False,
    ):
        """
        Configure Random Forest hyperparameters for cross-validation.

        :param n_estimators: Number of trees (try 100–500).
        :param max_depth: Maximum tree depth (None = unlimited; try 10–50).
        :param min_samples_leaf: Minimum samples at a leaf (try 5–20).
        :param max_features: Features per split — 'sqrt', 'log2', or a float.
        :param class_weight: 'balanced', 'balanced_subsample', or None.
        :param oob_score: Whether to compute out-of-bag accuracy.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.oob_score = oob_score

    # ------------------------------------------------------------------
    # Blocking
    # ------------------------------------------------------------------

    def create_blocks(self, n_blocks: int, axis: str = 'kmeans', method: str = 'quantile'):
        """
        Partition samples into spatial blocks for cross-validation.

        :param n_blocks: Number of blocks to create.
        :param axis: Split axis — 'x', 'y', 'z', or 'kmeans' for spatial clustering.
        :param method: 'quantile' (equal sample counts) or 'range' (equal coordinate width).
                       Only used when axis is 'x', 'y', or 'z'.
        """
        if axis == 'kmeans':
            block_labels = KMeans(n_clusters=n_blocks, random_state=42).fit(self.xyz).labels_
        elif axis in ('x', 'y', 'z'):
            axis_index = {'x': 0, 'y': 1, 'z': 2}[axis]
            coords = self.xyz[:, axis_index]
            block_labels = self._split_by_axis(coords, n_blocks, method)
        else:
            raise ValueError("axis must be 'x', 'y', 'z', or 'kmeans'.")

        self.clusters_feat = [self.features[block_labels == i] for i in range(n_blocks)]
        self.clusters_lab = [self.labels[block_labels == i] for i in range(n_blocks)]
        self.global_indices = [np.where(block_labels == i)[0] for i in range(n_blocks)]

        # Assign random colors per block for visualization
        colors = np.random.randint(0, 256, (n_blocks, 3))
        self.rgb_blocks = np.zeros((self.xyz.shape[0], 3))
        for i in range(self.rgb_blocks.shape[0]):
            self.rgb_blocks[i, :] = colors[block_labels[i], :]

    @staticmethod
    def _split_by_axis(coords: np.ndarray, n_blocks: int, method: str) -> np.ndarray:
        """Assign block labels by splitting a coordinate axis."""
        block_labels = np.zeros(len(coords), dtype=int)

        if method == 'quantile':
            thresholds = np.percentile(coords, np.linspace(0, 100, n_blocks + 1))
            for i in range(n_blocks):
                if i == n_blocks - 1:
                    mask = coords >= thresholds[i]
                else:
                    mask = (coords >= thresholds[i]) & (coords < thresholds[i + 1])
                block_labels[mask] = i

        elif method == 'range':
            min_c, max_c = coords.min(), coords.max()
            width = (max_c - min_c) / n_blocks
            for i in range(n_blocks):
                lower = min_c + i * width
                upper = min_c + (i + 1) * width
                if i == n_blocks - 1:
                    mask = coords >= lower
                else:
                    mask = (coords >= lower) & (coords < upper)
                block_labels[mask] = i
        else:
            raise ValueError("method must be 'quantile' or 'range'.")

        return block_labels

    # ------------------------------------------------------------------
    # Training & evaluation
    # ------------------------------------------------------------------

    def prepare_data_for_training(self, test_blocks_idx: list[int]):
        """
        Split block data into train and test sets.

        :param test_blocks_idx: Indices of blocks to use as the test set.
        :returns: (feat_train, lab_train, idx_train, feat_test, lab_test, idx_test)
        """
        n_blocks = len(self.clusters_feat)
        train_idx = [i for i in range(n_blocks) if i not in test_blocks_idx]

        feat_test = np.vstack([self.clusters_feat[i] for i in test_blocks_idx])
        feat_train = np.vstack([self.clusters_feat[i] for i in train_idx])

        lab_test = np.hstack([self.clusters_lab[i] for i in test_blocks_idx])
        lab_train = np.hstack([self.clusters_lab[i] for i in train_idx])

        indices_test = np.hstack([self.global_indices[i] for i in test_blocks_idx])
        indices_train = np.hstack([self.global_indices[i] for i in train_idx])

        return feat_train, lab_train, indices_train, feat_test, lab_test, indices_test

    def train_and_validate(self, model, n_test_blocks: int = 1, test_blocks_idx: list[int] | None = None):
        """
        Train a model and evaluate on held-out spatial blocks.

        :param model: A scikit-learn compatible classifier.
        :param n_test_blocks: Number of blocks to hold out (ignored if test_blocks_idx is set).
        :param test_blocks_idx: Explicit list of block indices to use as test set.
        :returns: (trained_model, train_accuracy, test_accuracy)
        """
        n_blocks = len(self.clusters_feat)

        if test_blocks_idx is None:
            test_blocks_idx = np.random.choice(n_blocks, size=n_test_blocks, replace=False).tolist()

        feat_train, lab_train, idx_train, feat_test, lab_test, idx_test = \
            self.prepare_data_for_training(test_blocks_idx)

        model.fit(feat_train, lab_train)

        acc_train, acc_test = self._calculate_accuracy(
            model, feat_train, lab_train, idx_train, feat_test, lab_test, idx_test
        )

        # Store OOB score if available
        if hasattr(model, 'oob_score_') and model.oob_score:
            self.last_oob_score = model.oob_score_

        # Store feature importances
        if hasattr(model, 'feature_importances_'):
            self.last_feature_importances = model.feature_importances_

        return model, acc_train, acc_test

    def _calculate_accuracy(self, model, feat_train, lab_train, idx_train, feat_test, lab_test, idx_test):
        """Compute train/test accuracy and store per-sample visualization colors."""
        # Colors: red = test-wrong, green = test-right, blue = train-wrong, cyan = train-right
        color_map = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]])
        rgb_validate = np.ones((self.n_samples, 3))

        pred_test = model.predict(feat_test)
        correct_test = (pred_test == lab_test)
        acc_test = correct_test.mean()

        for i, correct in enumerate(correct_test):
            rgb_validate[idx_test[i], :] = color_map[int(correct)]

        pred_train = model.predict(feat_train)
        correct_train = (pred_train == lab_train)
        acc_train = correct_train.mean()

        for i, correct in enumerate(correct_train):
            rgb_validate[idx_train[i], :] = color_map[int(correct) + 2]

        self.rgb_validate = rgb_validate
        return acc_train, acc_test

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def crossvalidate(
        self,
        n_blocks_list: list[int],
        n_models: int | None = None,
        n_test_blocks_list: list[int] | None = None,
        output_path: str = 'data.csv',
        blocking_method: str = 'kmeans',
    ):
        """
        Run block cross-validation across multiple block sizes.

        :param n_blocks_list: List of block counts to evaluate.
        :param n_models: Models per block size (default = n_blocks, i.e. leave-one-block-out).
        :param n_test_blocks_list: Test blocks per configuration (default: 1 each).
        :param output_path: CSV file to append results to.
        :param blocking_method: 'kmeans', 'horizontal', 'vertical_x', or 'vertical_y'.
        """
        if n_test_blocks_list is None:
            n_test_blocks_list = [1] * len(n_blocks_list)

        blocking_axis = {
            'kmeans': 'kmeans',
            'horizontal': 'z',
            'vertical_x': 'x',
            'vertical_y': 'y',
        }
        if blocking_method not in blocking_axis:
            raise ValueError(f"blocking_method must be one of {list(blocking_axis.keys())}.")

        loop1 = tqdm(
            zip(n_blocks_list, n_test_blocks_list),
            desc="Block sizes",
            position=0,
            leave=True,
            total=len(n_blocks_list),
        )
        loop2 = tqdm(desc="Models", position=1, leave=True)

        for n_blocks, n_test_blocks in loop1:
            self.create_blocks(n_blocks, axis=blocking_axis[blocking_method])

            n_model = n_models if n_models is not None else n_blocks
            loop2.reset(total=n_model)

            for i in range(n_model):
                model = create_random_forest(
                    n_features=self.n_features,
                    n_classes=self.n_classes,
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    class_weight=self.class_weight,
                    oob_score=self.oob_score,
                )

                model, acc_train, acc_test = self.train_and_validate(
                    model, n_test_blocks=1, test_blocks_idx=[i]
                )

                oob = self.last_oob_score if self.last_oob_score is not None else ''

                self._save_to_csv(
                    data=[
                        acc_train,
                        acc_test,
                        oob,
                        self.n_samples,
                        n_blocks,
                        n_test_blocks,
                        self.n_estimators,
                        self.max_depth if self.max_depth is not None else 'None',
                        self.min_samples_leaf,
                        self.max_features,
                        self.class_weight if self.class_weight is not None else 'None',
                        self.pca_applied,
                        blocking_method,
                        100 / n_blocks,
                        self.subset,
                        self.wave_lengths,
                        self.label_string,
                    ],
                    path=output_path,
                )
                loop2.update()

    # ------------------------------------------------------------------
    # CSV I/O
    # ------------------------------------------------------------------

    CSV_FIELDNAMES = [
        'acc_train', 'acc_test', 'oob_score', 'n_samples', 'n_blocks',
        'n_test_blocks', 'n_estimators', 'max_depth', 'min_samples_leaf',
        'max_features', 'class_weight', 'pca_applied',
        'blocking_method', 'block_percentage', 'subset',
        'wave_lengths', 'labels',
    ]

    def _save_to_csv(self, data: list, path: str):
        """Append one row of results to a CSV file."""
        print_header = not os.path.exists(path)
        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDNAMES)
            if print_header:
                writer.writeheader()
            writer.writerow(dict(zip(self.CSV_FIELDNAMES, data)))

    @staticmethod
    def load_from_csv(path: str = 'data.csv') -> list[dict]:
        """Load cross-validation results from a CSV file."""
        with open(path, 'r', newline='') as f:
            return list(csv.DictReader(f))

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_feature_importances(self, top_n: int = 20):
        """
        Bar chart of the top N most important features from the last trained model.

        :param top_n: Number of top features to display.
        """
        if self.last_feature_importances is None:
            raise RuntimeError("No model trained yet. Run train_and_validate first.")

        importances = self.last_feature_importances
        top_n = min(top_n, len(importances))
        indices = np.argsort(importances)[-top_n:][::-1]

        plt.figure(figsize=(12, 5))
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [f"Band {i}" for i in indices], rotation=45, ha='right')
        plt.xlabel('Feature (Band)')
        plt.ylabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        plt.show()

    def visualize_data(self, color_set: str, point_size: int = 3, legend: bool = False):
        """
        Interactive 3D scatter plot of the point cloud.

        :param color_set: 'validate', 'blocks', 'dataset', or 'labels'.
        :param point_size: Marker size.
        :param legend: Show train/test legend (only useful for 'validate').
        """
        color_map = {
            'validate': 'rgb_validate',
            'blocks': 'rgb_blocks',
            'dataset': 'rgb',
            'labels': 'rgb_labels',
        }
        if color_set not in color_map:
            raise ValueError(f"color_set must be one of {list(color_map.keys())}.")

        rgb = getattr(self, color_map[color_set]).astype(np.uint8)
        colors = [f"rgb({r},{g},{b})" for r, g, b in rgb]

        fig = plot.Figure(data=plot.Scatter3d(
            x=self.xyz[:, 0],
            y=self.xyz[:, 1],
            z=self.xyz[:, 2],
            mode='markers',
            showlegend=False,
            marker=dict(size=point_size, color=colors, opacity=1),
        ))

        if legend:
            legend_items = [
                ("test — wrong", "rgb(255,0,0)"),
                ("test — right", "rgb(0,255,0)"),
                ("train — wrong", "rgb(0,0,255)"),
                ("train — right", "rgb(0,255,255)"),
            ]
            for name, color in legend_items:
                fig.add_trace(plot.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode="markers",
                    marker=dict(size=6, color=color),
                    name=name,
                    showlegend=True,
                ))

        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
            ),
            width=900,
            height=700,
        )
        fig.show()

    @staticmethod
    def plot_data(data: list[dict], y_axis: str = 'acc_test'):
        """
        Box plot of accuracy vs. block size from saved CSV results.

        :param data: List of result dicts (from load_from_csv).
        :param y_axis: Column to plot on the y-axis (default: 'acc_test').
        """
        print(f"Plotting {len(data)} data points as boxplot...")
        x_plot = [100 / int(d['n_blocks']) for d in data]
        y_plot = [100 * float(d[y_axis]) for d in data]

        box_sizes = np.unique(x_plot)
        plt.boxplot(
            [[y for x, y in zip(x_plot, y_plot) if x == bs] for bs in box_sizes]
        )
        plt.xticks(range(1, len(box_sizes) + 1), [f"{bs:.2f}%" for bs in box_sizes])
        plt.xlabel('Block size')
        plt.ylabel('Accuracy (%)')
        plt.title(f'{y_axis} by block size')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # PyTorch Dataset (for future neural network experiments)
    # ------------------------------------------------------------------

    class TintoDataset(Dataset):
        """PyTorch Dataset wrapper for Tinto features and labels."""

        def __init__(self, X, Y, indices):
            self.X = X
            self.Y = Y
            self.indices = indices

        def __getitem__(self, index):
            return self.X[index], self.Y[index], self.indices[index]

        def __len__(self):
            return len(self.X)
