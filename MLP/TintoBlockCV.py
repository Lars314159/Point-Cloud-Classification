from hylite import io
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from sklearn.cluster import KMeans
import plotly.graph_objects as plot
from tqdm.notebook import tqdm
import csv
import matplotlib.pyplot as plt
from typing import Optional, Sequence, List, Callable, Any

class TintoBlockCV:
    def load_dataset(self,
                     path : str,
                     subset : str,
                     wave_lengths : str,
                     labels : str = 'complete',
                     n_samples: int | None = None,
                     normalize : bool = True) -> None:
        """
        Loads the specified data from the Tinto dataset.

        :param path: Path to the Tinto '.hyc' folder.
        :type path: str
        :param subset: One of 'real', 'synthetic' or 'degraded'.
        :type subset: str
        :param wave_lengths: One of 'VNIR', 'SWIR' or 'LWIR'.
        :type wave_lengths: str
        :param labels: 'basic' or 'complete' (default 'complete').
        :type labels: str
        :param n_samples: Number of samples to draw randomly from the dataset. If None use all samples.
        :type n_samples: int | None
        :param normalize: If True (default) the dataset will be z-score normalized in-place.
        :type normalize: bool
        """
        if (not n_samples is None) and n_samples <= 0:
            raise ValueError("n_samples has to be a positive integer or None")
        
        dataset = io.load(os.path.abspath(path))
        self.class_names = dataset.labels_complete.header.get_list('class names')
        
        match labels:
            case "basic":
                lab = dataset.labels_basic.data[:,0]      
            case "complete":
                lab = dataset.labels_complete.data[:,0]      
            case _:
                raise ValueError("labels has to be 'basic' or 'complete'")
        
        match subset:
            case "real":
                data = dataset.real
            case "synthetic":
                data = dataset.synth
            case "degraded":
                data = dataset.degr
            case _:
                raise ValueError("subset has to be 'real', 'synthetic' or 'degraded'")

        match wave_lengths:
            case "VNIR":
                data = data.vnir
            case "SWIR":
                data = data.swir
            case "LWIR":
                data = data.lwir
            case _:
                raise ValueError("wave_lengths has to be 'VNIR', 'SWIR' or 'LWIR'")

        features = data.data
        xyz = data.xyz
        rgb = data.rgb
        n_total_samples = features.shape[0]
        
        if not n_samples is None:
            if n_samples > n_total_samples:
                raise ValueError(f"n_samples is too big, it must not be bigger than {n_total_samples}")
            idx = np.random.choice(range(n_total_samples),size=n_samples)
        else:
            idx = np.array(range(n_total_samples))
            np.random.shuffle(idx)

        self.features = features[idx,:]
        self.labels = lab[idx].astype(np.int64)
        self.xyz = xyz[idx]
        self.rgb = rgb[idx]
        self.n_samples = self.features.shape[0]
        self.n_features = self.features.shape[1]
        self.n_classes = len(self.class_names)
        self.subset = subset
        self.label_string = labels
        self.wave_lengths = wave_lengths
        
        self.generate_label_rgb()
        if normalize:
            self.normalize_data()

    def generate_label_rgb(self) -> None:
        """
        Generate an RGB color mapping for every sample based on its label.
        """
        
        if not hasattr(self, 'labels') or not hasattr(self, 'n_classes'):
            raise ValueError("labels and n_classes must be set (call load_dataset first)")
        if self.n_classes <= 0:
            raise ValueError("n_classes must be a positive integer")
        
        colors = np.random.randint(0,256,(self.n_classes,3))
        self.rgb_labels = np.zeros((self.n_samples,3))
        for i in range(self.n_samples):
            self.rgb_labels[i,:] = colors[self.labels[i],:]

    
    def normalize_data(self) -> None:
        """
        Z-score normalize the feature matrix stored in ``self.features`` (in-place).
        """
        if not hasattr(self, 'features'):
            raise ValueError("features not found. Call load_dataset first.")
        
        mean = self.features.mean(axis=0)
        std = self.features.std(axis=0)
        self.features = (self.features - mean) / std

    class TintoDataset(Dataset):
        """Stores features and labels, necessary for PyTorch DataLoader.

        :param X: Feature tensor of shape (n_samples, n_features).
        :type X: torch.Tensor
        :param Y: Label tensor of shape (n_samples,).
        :type Y: torch.Tensor
        :param indices: Original global indices of the samples (used for mapping results back).
        :type indices: Sequence[int]
        """
        
        def __init__(self, X: torch.Tensor, Y: torch.Tensor, indices: Sequence[int]):
            self.X = X
            self.Y = Y
            self.indices = indices
            
        def __getitem__(self, index: int) -> tuple:
            return self.X[index], self.Y[index], self.indices[index]
        
        def __len__(self) -> int:
            return len(self.X)
        
    def set_model_parameters(self, 
                             architecture: Callable[[], Any], 
                             learning_rate: float, 
                             batch_size: int) -> None:
        """
        Sets basic parameters required for training.

        :param architecture: A callable returning a model instance when called.
        :type architecture: callable
        :param learning_rate: Learning rate for the optimizer (must be positive).
        :type learning_rate: float
        :param batch_size: Batch size used in the DataLoader (must be a positive integer).
        :type batch_size: int
        """
        
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
    def create_blocks(self, n_blocks: int) -> None:
        """
        Creates spatial blocks using KMeans on the xyz coordinates.

        :param n_blocks: Number of spatial blocks/clusters to create.
        :type n_blocks: int
        """
        
        kmeans = KMeans(n_clusters=n_blocks).fit(self.xyz).labels_
        self.clusters_feat = [self.features[kmeans == i] for i in range(n_blocks)]
        self.clusters_lab = [self.labels[kmeans == i] for i in range(n_blocks)]
        self.global_indices = [np.where(kmeans == i)[0] for i in range(n_blocks)]

        colors = np.random.randint(0,256,(n_blocks,3))
        self.rgb_blocks = np.zeros((self.xyz.shape[0],3))
        for i in range(self.rgb_blocks.shape[0]):
            self.rgb_blocks[i,:] = colors[kmeans[i],:]

    def prepare_data_for_training(self, test_blocks_idx: Sequence[int]) -> tuple[DataLoader, DataLoader]:
        """
        Prepares PyTorch DataLoaders for training and testing given indices of test blocks.

        :param test_blocks_idx: Indices of clusters to be used as the test set.
        :type test_blocks_idx: Sequence[int]
        :returns: Dataloader of training data, Dataloader for test data
        :rtype: Tuple[DataLoader, DataLoader]
        """
        if not hasattr(self, 'clusters_feat'):
            raise ValueError("Clusters not created. Call create_blocks first.")
        n_blocks = len(self.clusters_feat)

        if len(test_blocks_idx) == 0:
            raise ValueError("test_blocks_idx must contain at least one block index")
        for idx in test_blocks_idx:
            if idx < 0 or idx >= n_blocks:
                raise ValueError(f"Each test block index must be an int in [0, {n_blocks-1}]")
        
        feat_test = np.vstack([self.clusters_feat[i] for i in test_blocks_idx])
        feat_train = np.vstack([self.clusters_feat[i] for i in range(n_blocks) if not i in test_blocks_idx])
        lab_test = np.hstack([self.clusters_lab[i] for i in test_blocks_idx])
        lab_train = np.hstack([self.clusters_lab[i] for i in range(n_blocks) if not i in test_blocks_idx])
        indices_test = np.hstack([self.global_indices[i] for i in test_blocks_idx])
        indices_train = np.hstack([self.global_indices[i] for i in range(n_blocks) if not i in test_blocks_idx])

        feat_train = torch.tensor(feat_train, dtype=torch.float32)
        feat_test = torch.tensor(feat_test, dtype=torch.float32)
        lab_train = torch.tensor(lab_train, dtype=torch.int64)
        lab_test = torch.tensor(lab_test, dtype=torch.int64)

        dataset_train = self.TintoDataset(feat_train, lab_train, indices_train)
        dataset_test = self.TintoDataset(feat_test, lab_test, indices_test)
        dataloader_train = DataLoader(dataset=dataset_train, batch_size=self.batch_size, shuffle=False)
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=self.batch_size, shuffle=False)
        
        return dataloader_train, dataloader_test
        
    def train_and_validate(self, 
                           model: torch.nn.Module,
                           n_epochs: int, 
                           n_test_blocks: int = 1, 
                           test_blocks_idx: Optional[Sequence[int]] = None) -> tuple[torch.nn.Module, float, float]:
        """
        Trains the provided model for n_epochs and evaluates it on a held-out block.

        :param model: An instantiated PyTorch model.
        :type model: torch.nn.Module
        :param n_epochs: Number of epochs to train.
        :type n_epochs: int
        :param n_test_blocks: Number of blocks to select as test blocks if test_blocks_idx is None.
        :type n_test_blocks: int
        :param test_blocks_idx: If provided, explicit indices of blocks to use as the test set.
        :type test_blocks_idx: Sequence[int] | None

        :returns: model, training accuracy, test accuracy
        :rtype: Tuple[torch.nn.Module, float, float]
        """
        
        if n_epochs <= 0:
            raise ValueError("n_epochs must be a positive integer")
        if not hasattr(self, 'clusters_feat'):
            raise ValueError("Blocks not created. Call create_blocks first.")

        n_blocks = len(self.clusters_feat)
        if test_blocks_idx is None:
            if n_test_blocks <= 0:
                raise ValueError("n_test_blocks must be a positive integer")
            if n_test_blocks > n_blocks:
                raise ValueError("n_test_blocks cannot be larger than number of blocks")
        else:
            for idx in test_blocks_idx:
                if idx < 0 or idx >= n_blocks:
                    raise ValueError(f"Each test block index must be an int in [0, {n_blocks-1}]")
        
        if test_blocks_idx is None:
            test_blocks_idx = np.random.choice(range(n_blocks),size=n_test_blocks)
        
        dataloader_train, dataloader_test = self.prepare_data_for_training(test_blocks_idx)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=self.learning_rate)    
        outer_loop = tqdm(range(n_epochs), desc="epochs", position=0, total=n_epochs, leave=True)
        inner_loop = tqdm(desc="iterations", position=1, leave=True, mininterval=0.1)   

        for epoch in outer_loop:
            inner_loop.reset(total=len(dataloader_train))
            for inputs,labels,_ in dataloader_train:
                optimizer.zero_grad()
                y_pred = model(inputs)
                lo = loss(y_pred,labels)
                lo.backward()
                optimizer.step()
                inner_loop.update()   
        
        acc_train, acc_test = self.calculate_accuracy(model, dataloader_train, dataloader_test)
        return model, acc_train, acc_test
        
    def calculate_accuracy(self, 
                           model: torch.nn.Module, 
                           dataloader_train: DataLoader, 
                           dataloader_test: DataLoader) -> tuple[float, float]:
        """
        Calculates per-split accuracy and stores a validation RGB map matching the
        original sample indices.

        :param model: model used for predictions.
        :type model: torch.nn.Module
        :param dataloader_train: DataLoader for training samples.
        :type dataloader_train: DataLoader
        :param dataloader_test: DataLoader for test samples.
        :type dataloader_test: DataLoader

        :returns: training accuracy, test accuracy as floats in the [0,1] range.
        :rtype: Tuple[float, float]
        """
        
        with torch.no_grad():
            n_total_test = 0
            for inputs, _, _ in dataloader_test:
                n_total_test += len(inputs)
            n_total_train = 0        
            for inputs, _, _ in dataloader_train:
                n_total_train += len(inputs)
            
            if n_total_test == 0 or n_total_train == 0:
                raise ValueError("Train and test loaders must not be empty")
                
            colors = np.array([[255,0,0],[0,255,0],[0,0,255],[0,255,255]])
            rgb_validate = np.ones((n_total_test+n_total_train,3))
            acc_test = 0
            for inputs, labels, index in dataloader_test:
                y_pred = model(inputs)
                _, pred_labels = torch.max(y_pred,1)
                correct_prediction = pred_labels == labels
                acc_test += sum(correct_prediction) 
                for i in range(len(labels)):
                    rgb_validate[index[i],:] = colors[correct_prediction[i],:]
            acc_test = acc_test / n_total_test
            
            acc_train = 0
            for inputs, labels, index in dataloader_train:
                y_pred = model(inputs)
                _, pred_labels = torch.max(y_pred,1)
                correct_prediction = pred_labels == labels
                acc_train += sum(correct_prediction) 
                for i in range(len(labels)):
                    rgb_validate[index[i],:] = colors[correct_prediction[i]+2,:]
            acc_train = acc_train / n_total_train
            self.rgb_validate = rgb_validate      
                  
            return acc_train.item(), acc_test.item()
        
    def visualize_data(self, 
                       color_set: str, 
                       point_size: float = 3, 
                       legend: bool = False) -> None:
        """
        Visualize the point cloud (xyz) colored by one of several predefined color sets.

        :param color_set: One of 'validate', 'blocks', 'original', 'labels'.
        :type color_set: str
        :param point_size: Size of the points in the plot. Default is 3.
        :type point_size: float
        :param legend: Whether to show a legend (default False).
        :type legend: bool
        """
        
        match color_set:
            case 'validate':
                rgb = getattr(self, 'rgb_validate', None)
                if rgb is None:
                    raise ValueError("Validation RGB not available. Run calculate_accuracy/train_and_validate first.")
            case 'blocks':
                rgb = getattr(self, 'rgb_blocks', None)
                if rgb is None:
                    raise ValueError("Block RGB not available. Run create_blocks first.")
            case 'original':
                rgb = getattr(self, 'rgb', None)
                if rgb is None:
                    raise ValueError("Original RGB not found. Run load_dataset first.")
            case 'labels':
                rgb = getattr(self, 'rgb_labels', None)
                if rgb is None:
                    raise ValueError("Label RGB not found. Run load_dataset first.")
            case _:
                raise ValueError("color_set must be one of 'validate', 'blocks', 'dataset', 'labels'")
        
        rgb = rgb.astype(np.uint)
        colors = [f"rgb({r},{g},{b})" for r, g, b in rgb]
        
        fig = plot.Figure(data=plot.Scatter3d(
            x=self.xyz[:, 0],
            y=self.xyz[:, 1],
            z=self.xyz[:, 2],
            mode='markers',
            showlegend=False,
            marker=dict(
                size=point_size,
                color=colors,
                opacity=1
        )))
        if legend:
            fig.add_trace(plot.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                marker=dict(size=6, color="rgb(255,0,0)"),
                name="test - wrong",
                showlegend=True
            ))
            fig.add_trace(plot.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                marker=dict(size=6, color="rgb(0,255,0)"),
                name="test - right",
                showlegend=True
            ))
            fig.add_trace(plot.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                marker=dict(size=6, color="rgb(0,0,255)"),
                name="training - wrong",
                showlegend=True
            ))
            fig.add_trace(plot.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                marker=dict(size=6, color="rgb(0,255,255)"),
                name="training - right",
                showlegend=True
            ))

        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data"
            ),
            width=900,
            height=700,
        )

        fig.show()

    def crossvalidate(self, 
                      n_epochs: int, 
                      n_blocks_list: Sequence[int], 
                      n_models: Optional[int] = None, 
                      output_path: str = 'data.csv') -> None:
        """
        Runs crossvalidation for each of the specified block sizes.

        :param n_epochs: Number of training epochs to run for each model.
        :type n_epochs: int
        :param n_blocks_list: Sequence of different block sizes to use.
        :type n_blocks_list: Sequence[int]
        :param n_models: Number of different models to train per block size.
        If None, it defaults to the maximum value in n_blocks_list.
        :type n_models: int | None
        :param output_path: CSV file path where results are appended.
        :type output_path: str
        """
        if n_epochs <= 0:
            raise ValueError("n_epochs must be a positive integer")
        if len(n_blocks_list) == 0:
            raise ValueError("n_blocks_list must be a non-empty sequence of integers")
        if n_models is not None and n_models <= 0:
            raise ValueError("n_models must be a positive integer or None")
        if not hasattr(self, 'architecture'):
            raise ValueError("Model architecture not set. Call set_model_parameters first.")
        
        loop1 = tqdm(n_blocks_list,  desc="block sizes", position=0, leave=True, total=len(n_blocks_list), smoothing=0.1)
        loop2 = tqdm(desc="models", position=1, leave=True, total=n_models, smoothing=0.1)
        outer_loop = tqdm(desc="epochs", position=2, leave=True, total=n_epochs, smoothing=0.1)
        inner_loop = tqdm(desc="iterations", position=3, leave=True, mininterval=0.01, smoothing=0.1)   
        loss = nn.CrossEntropyLoss()

        for n_blocks in loop1:
            if n_models is None:
                n_model = n_blocks
            self.create_blocks(n_blocks)
            loop2.reset(total=n_model)
            for i in range(n_model):
                model = self.architecture()
                dataloader_train, dataloader_test = self.prepare_data_for_training(test_blocks_idx=[i % n_blocks])
                optimizer = torch.optim.Adam(model.parameters(),lr=self.learning_rate)    
    
                outer_loop.reset()
                for epoch in range(n_epochs):
                    inner_loop.reset(total=len(dataloader_train))
                    for inputs,labels,_ in dataloader_train:
                        optimizer.zero_grad()
                        y_pred = model(inputs)
                        lo = loss(y_pred,labels)
                        lo.backward()
                        optimizer.step()
                        inner_loop.update()   
                    outer_loop.update() 
                
                acc_train, acc_test = self.calculate_accuracy(model, dataloader_train, dataloader_test)                 
                self.save_to_csv([acc_train, 
                                  acc_test, 
                                  self.n_samples, 
                                  n_blocks,  
                                  n_epochs, 
                                  self.batch_size, 
                                  self.learning_rate, 
                                  self.subset,
                                  self.wave_lengths,
                                  self.label_string], output_path)
                loop2.update()

    def save_to_csv(self, 
                    data: Sequence[Any], 
                    path: str) -> None:
        """
        Append a row of results to a CSV file. If the file does not exist, a header is
        written first.

        :param data: Data values in the same order as the fieldnames in this method.
        :type data: Sequence[Any]
        :param path: Output CSV file path.
        :type path: str
        """
        
        print_header = not os.path.exists(path)
        with open(path,'a',newline='') as file:
            fieldnames = ['acc_train', 'acc_test', 'n_samples', 'n_blocks', 'n_epochs', 'batch_size', 'learning_rate', 'subset', 'wave_lengths', 'labels']
            writer = csv.DictWriter(file, fieldnames)
            if print_header:
                writer.writeheader()               
            writer.writerow(dict(zip(fieldnames,data)))
        
    @staticmethod
    def load_from_csv(path: str = 'data.csv') -> List[dict]:
        """
        Load results from a CSV previously written by save_to_csv.

        :param path: Path to the CSV file.
        :type path: str

        :returns: list of dict, each dict corresponds to a CSV row with string values.
        :rtype: List[dict]
        """
        
        with open(path,'r',newline='') as file:
            reader = csv.DictReader(file)
            data = []
            for row in reader:
                data.append(row)
            
            return data


    @staticmethod
    def plot_data(data: Sequence[dict]) -> None:
        """
        Plot a boxplot of accuracies stored in the CSV file.

        :param data: Iterable of rows as returned by ``load_from_csv``.
        :type data: Sequence[dict]
        """
        
        x_plot = []
        y_plot_test = []
        y_plot_train = []
        for data_point in data:
            x_plot.append(100/int(data_point['n_blocks']))
            y_plot_test.append(100*float(data_point['acc_test']))
            y_plot_train.append(100*float(data_point['acc_train']))
        # plt.plot(x_plot,y_plot,'r.')
        
        box_sizes = np.unique(x_plot)
        x = 2*np.arange(len(box_sizes))
        offset = 0.25
        
        fig, ax = plt.subplots()
        boxplt1 = ax.boxplot([[y for (x,y) in zip(x_plot, y_plot_train) if x==box_size] for box_size in box_sizes],positions=x-offset,patch_artist=True,label='Train')
        boxplt2 = ax.boxplot([[y for (x,y) in zip(x_plot, y_plot_test) if x==box_size] for box_size in box_sizes],positions=x+offset,patch_artist=True,label='Test')
        for box in boxplt1["boxes"]:
            box.set_facecolor("blue")
        for box in boxplt2["boxes"]:
            box.set_facecolor("orange")
        plt.xticks(range(0,2*len(box_sizes),2),[f"{box_size:.2f}%" for box_size in box_sizes])
        plt.xlabel('block sizes')
        plt.ylabel('accuracy (%)')
        plt.legend()
    
def main():
    # example 1: crossvalidation
    from TintoBlockCV import TintoBlockCV
    from model import MLP
    from functools import partial

    TBCV = TintoBlockCV()
    TBCV.load_dataset(path='Tinto/tinto3D.hyc/',
                    subset='synthetic',
                    wave_lengths='VNIR',
                    n_samples=50_000)

    architecture = partial(MLP, 
                        n_features = TBCV.n_features,  
                        n_classes = TBCV.n_classes,
                        n_neurons = 100, 
                        n_hidden_layers = 3)

    TBCV.set_model_parameters(architecture=architecture,
                            learning_rate=1e-4,
                            batch_size=32)

    TBCV.crossvalidate(n_epochs=10,
                    n_blocks_list=[5, 10, 20, 30, 40, 50],
                    output_path='data5.csv')
    
    #example 2: training a single model and visualizing the data
    from TintoBlockCV import TintoBlockCV
    from model import MLP
    from functools import partial
    TBCV = TintoBlockCV()
    TBCV.load_dataset(path='Tinto/tinto3D.hyc/',
                    n_samples=50_000,
                    subset='real',
                    wave_lengths='VNIR',
                    labels='complete')
    TBCV.visualize_data(color_set='dataset')
    TBCV.visualize_data(color_set='labels')
    TBCV.create_blocks(n_blocks=10)
    TBCV.visualize_data(color_set='blocks')
    architecture = partial(MLP, 
                       n_features = TBCV.n_features,  
                       n_classes = TBCV.n_classes,
                       n_neurons = 100, 
                       n_hidden_layers = 3)

    TBCV.set_model_parameters(architecture=architecture,
                            learning_rate=1e-4,
                            batch_size=32)

    model_instance = architecture()
    print(model_instance)
    model_instance, acc_train, acc_test = TBCV.train_and_validate(model=model_instance,
                                                              n_epochs=10,
                                                              n_test_blocks=1)
    print(f"training accuracy: {acc_train:.2%}, test accuracy: {acc_test:.2%}")
    TBCV.visualize_data('validate', legend=True)