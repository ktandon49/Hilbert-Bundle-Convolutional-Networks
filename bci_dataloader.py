"""
PyTorch Geometric DataLoader for BCI Competition IV Datasets 2a and 2b

Represents EEG data as graphs where:
- Nodes = EEG channels/electrodes
- Edges = Spatial connectivity between electrodes
- Node features = Time-series signals from each channel

Requirements:
pip install torch-geometric mne numpy torch scipy
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import mne
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
from typing import Tuple, Optional, List, Dict


class BCIGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset for BCI Competition IV datasets 2a and 2b
    
    Each EEG trial is converted to a graph:
    - Nodes: EEG channels
    - Node features: Time-series data from each channel
    - Edges: Spatial connectivity based on electrode positions
    
    Args:
        data_path: Path to directory containing .gdf files
        dataset_type: '2a' or '2b'
        subjects: List of subject IDs
        train: Whether to load training or evaluation data
        tmin: Start time of epoch (seconds)
        tmax: End time of epoch (seconds)
        baseline: Baseline correction period
        graph_type: 'distance' (k-nearest neighbors) or 'fully_connected'
        k_neighbors: Number of nearest neighbors for distance-based graphs
        normalize: Whether to apply z-score normalization
    """
    
    def __init__(
        self,
        data_path: str,
        dataset_type: str = '2a',
        subjects: Optional[List[str]] = None,
        train: bool = True,
        tmin: float = 0.5,
        tmax: float = 4.5,
        baseline=None,
        graph_type: str = 'distance',  # 'distance' or 'fully_connected'
        k_neighbors: int = 4,
        normalize: bool = True
    ):
        super().__init__()
        
        self.data_path = Path(data_path)
        self.dataset_type = dataset_type
        self.train = train
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.graph_type = graph_type
        self.k_neighbors = k_neighbors
        self.normalize = normalize
        
        # Define class labels and subjects
        if dataset_type == '2a':
            self.n_classes = 4
            # Use hexadecimal format for event IDs
            self.event_id = {'left': 0x0301, 'right': 0x0302, 'foot': 0x0303, 'tongue': 0x0304}
            self.class_names = ['left', 'right', 'foot', 'tongue']
            if subjects is None:
                subjects = [f'A0{i}' for i in range(1, 10)]
            
            # Standard 10-20 electrode positions for dataset 2a
            self.bci_pos = [
                'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                'P3', 'P1', 'Pz', 'P2', 'P4'
            ]
        else:  # 2b
            self.n_classes = 2
            self.event_id = {'left': 769, 'right': 770}
            self.class_names = ['left', 'right']
            if subjects is None:
                subjects = [f'B0{i}' for i in range(1, 10)]
            
            # Electrode positions for dataset 2b
            self.bci_pos = ['C3', 'Cz', 'C4']
        
        self.subjects = subjects
        
        # Load data
        self._load_data()
        
        # Create graph structure based on electrode positions
        self.edge_index, self.edge_attr = self._create_graph_structure()
        
        print(f"\n{'='*60}")
        print(f"Loaded BCI Competition IV Dataset {dataset_type} (PyG Format)")
        print(f"{'='*60}")
        print(f"Total trials: {len(self.data_list)}")
        print(f"Number of nodes (channels): {self.num_nodes}")
        print(f"Number of edges: {self.edge_index.shape[1]}")
        print(f"Node feature dimension: {self.data_list[0].x.shape[1]}")
        print(f"Classes: {self.n_classes} - {self.class_names}")
        print(f"Class distribution: {np.bincount([d.y.item() for d in self.data_list])}")
        print(f"{'='*60}\n")
    
    def _load_data(self):
        """Load all EEG data and store as list of graphs"""
        self.data_list = []
        self.electrode_positions = None
        self.channel_names = None
        
        for subject in self.subjects:
            print(f"\nLoading subject {subject}...")
            graphs, positions, ch_names = self._load_subject(subject)
            self.data_list.extend(graphs)
            
            # Store electrode info from first subject
            if self.electrode_positions is None:
                self.electrode_positions = positions
                self.channel_names = ch_names
        
        self.num_nodes = len(self.channel_names)
    
    def _get_electrode_positions_2a(self) -> Dict[str, np.ndarray]:
        """
        Get 3D positions for 22 EEG channels in dataset 2a
        Based on standard 10-20 system
        """
        # Standard 10-20 positions (approximate Cartesian coordinates)
        # These are normalized positions on a unit sphere
        positions_10_20 = {
            'Fz': np.array([0, 0.5, 0.7]),
            'FC3': np.array([-0.4, 0.3, 0.6]),
            'FC1': np.array([-0.2, 0.3, 0.7]),
            'FCz': np.array([0, 0.3, 0.7]),
            'FC2': np.array([0.2, 0.3, 0.7]),
            'FC4': np.array([0.4, 0.3, 0.6]),
            'C5': np.array([-0.6, 0, 0.5]),
            'C3': np.array([-0.5, 0, 0.6]),
            'C1': np.array([-0.2, 0, 0.7]),
            'Cz': np.array([0, 0, 0.7]),
            'C2': np.array([0.2, 0, 0.7]),
            'C4': np.array([0.5, 0, 0.6]),
            'C6': np.array([0.6, 0, 0.5]),
            'CP3': np.array([-0.4, -0.3, 0.6]),
            'CP1': np.array([-0.2, -0.3, 0.7]),
            'CPz': np.array([0, -0.3, 0.7]),
            'CP2': np.array([0.2, -0.3, 0.7]),
            'CP4': np.array([0.4, -0.3, 0.6]),
            'P3': np.array([-0.4, -0.5, 0.5]),
            'P1': np.array([-0.2, -0.5, 0.6]),
            'Pz': np.array([0, -0.5, 0.7]),
            'P2': np.array([0.2, -0.5, 0.6]),
            'P4': np.array([0.4, -0.5, 0.5]),
        }
        return positions_10_20
    
    def _get_electrode_positions_2b(self) -> Dict[str, np.ndarray]:
        """Get positions for 3 channels in dataset 2b (C3, Cz, C4)"""
        return {
            'C3': np.array([-0.5, 0, 0.6]),
            'Cz': np.array([0, 0, 0.7]),
            'C4': np.array([0.5, 0, 0.6]),
        }
    
    def _create_graph_structure(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Create edge_index and edge_attr based on electrode positions
        
        Returns:
            edge_index: (2, num_edges) - connectivity
            edge_attr: (num_edges, 1) - edge weights (distances)
        """
        # Get electrode positions using bci_pos ordering
        positions_array = np.array([
            self.electrode_positions[self.bci_pos[i]] 
            for i, _ in enumerate(self.channel_names)
        ])
        
        if self.graph_type == 'fully_connected':
            # Create fully connected graph
            num_nodes = len(self.channel_names)
            edge_index = torch.tensor([
                [i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j
            ], dtype=torch.long).t()
            
            # Edge weights are inverse distances
            distances = squareform(pdist(positions_array, metric='euclidean'))
            edge_weights = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        edge_weights.append(1.0 / (distances[i, j] + 1e-6))
            
            edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
            
        else:  # 'distance' - k-nearest neighbors
            # Compute pairwise distances
            distances = squareform(pdist(positions_array, metric='euclidean'))
            
            # Create k-NN graph
            edge_list = []
            edge_weights = []
            
            for i in range(len(self.channel_names)):
                # Get k nearest neighbors (excluding self)
                k_nearest = np.argsort(distances[i])[1:self.k_neighbors+1]
                
                for j in k_nearest:
                    edge_list.append([i, j])
                    # Edge weight is inverse distance
                    edge_weights.append(1.0 / (distances[i, j] + 1e-6))
            
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        
        # Make graph undirected by adding reverse edges
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        
        # Remove duplicate edges
        edge_index, edge_attr = self._remove_duplicate_edges(edge_index, edge_attr)
        
        return edge_index, edge_attr
    
    def _remove_duplicate_edges(self, edge_index, edge_attr):
        """Remove duplicate edges while keeping edge attributes"""
        # Create unique edge identifier
        edge_ids = edge_index[0] * self.num_nodes + edge_index[1]
        unique_edges, inverse_indices = torch.unique(edge_ids, return_inverse=True)
        
        # Keep first occurrence of each edge
        unique_mask = torch.zeros(len(edge_ids), dtype=torch.bool)
        unique_mask[torch.unique(inverse_indices, return_inverse=True)[1]] = True
        
        return edge_index[:, unique_mask], edge_attr[unique_mask]
    
    def _load_subject(self, subject: str) -> Tuple[List[Data], Dict, List[str]]:
        """Load and convert subject data to graph format"""
        # Construct filename
        suffix = 'T' if self.train else 'E'
        filename = f"{subject}{suffix}.gdf"
        filepath = self.data_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Load GDF file
        raw = mne.io.read_raw_gdf(str(filepath), preload=True, verbose=False)
        
        # Get EEG channels
        if self.dataset_type == '2a':
            eeg_channels = [ch for ch in raw.ch_names if ch.startswith('EEG-')][:22]
            # Remove 'EEG-' prefix for matching with position dict
            channel_names = [ch.replace('EEG-', '') for ch in eeg_channels]
            positions = self._get_electrode_positions_2a()
        else:  # 2b
            eeg_channels = ['EEG-C3', 'EEG-Cz', 'EEG-C4']
            channel_names = ['C3', 'Cz', 'C4']
            positions = self._get_electrode_positions_2b()
        
        raw.pick_channels(eeg_channels)
        
        # Apply bandpass filter
        raw.filter(l_freq=4.0, h_freq=40.0, verbose=False)
        
        # Find events and convert IDs
        events, event_hash = mne.events_from_annotations(raw, verbose=False)
        
        # Get valid event IDs
        valid_event_ids = list(self.event_id.values())
        
        # Convert event IDs using MNE's hash mapping
        conv_valid_event_ids = [event_hash[k] for k in event_hash if int(k) in valid_event_ids]
        
        print(f"  Original event IDs: {valid_event_ids}")
        print(f"  Converted event IDs: {conv_valid_event_ids}")
        
        # Filter events to keep only motor imagery
        mask = np.isin(events[:, 2], conv_valid_event_ids)
        events = events[mask]
        
        print(f"  Found {len(events)} motor imagery trials")
        print(f"  Time window: tmin={self.tmin}, tmax={self.tmax}")
        print(f"  Baseline: {self.baseline}")
        
        # Create epochs
        epochs = mne.Epochs(
            raw, events,
            event_id=conv_valid_event_ids,
            tmin=self.tmin, tmax=self.tmax,
            baseline=self.baseline,
            preload=True, verbose=False, proj=False,
            on_missing='warn'  # Don't fail on missing events
        )
        
        # Get data: (n_epochs, n_channels, n_times)
        X = epochs.get_data()
        
        # Get labels
        event_id_to_class = {eid: i for i, eid in enumerate(conv_valid_event_ids)}
        y = np.array([event_id_to_class[event[2]] for event in events])
        
        print(f"  Class distribution: {np.bincount(y)}")
        
        # Normalize if requested
        if self.normalize:
            X = (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-8)
        
        # Convert to list of PyG Data objects
        graph_list = []
        for i in range(len(X)):
            # Node features: (num_nodes, num_timepoints)
            node_features = torch.FloatTensor(X[i])
            
            # Create PyG Data object
            graph = Data(
                x=node_features,  # (num_nodes, num_features=time_steps)
                y=torch.LongTensor([y[i]]),
                num_nodes=len(channel_names)
            )
            
            graph_list.append(graph)
        
        print(f"  Created {len(graph_list)} graphs")
        
        return graph_list, positions, channel_names
    
    def len(self) -> int:
        return len(self.data_list)
    
    def get(self, idx: int) -> Data:
        """
        Returns a PyG Data object with:
            - x: (num_nodes, num_features) node features
            - edge_index: (2, num_edges) graph connectivity
            - edge_attr: (num_edges, 1) edge weights
            - y: (1,) class label
        """
        data = self.data_list[idx].clone()
        
        # Add graph structure
        data.edge_index = self.edge_index
        data.edge_attr = self.edge_attr
        
        return data


class EEG_GNN(torch.nn.Module):
    """
    Graph Neural Network for EEG motor imagery classification
    
    Architecture:
    - Temporal convolution to capture time dependencies
    - Graph convolution layers to capture spatial relationships
    - Global pooling and classification
    """
    
    def __init__(
        self,
        in_channels: int,  # This will be the number of time points
        hidden_channels: int = 64,
        num_classes: int = 4,
        num_graph_layers: int = 3,
        dropout: float = 0.5,
        graph_conv: str = 'GCN'  # 'GCN' or 'GAT'
    ):
        super().__init__()
        
        # Temporal feature extraction (1D conv over time)
        # Input: (batch, 1, time_steps) - treating each node's signal as 1 channel
        self.temporal_conv = torch.nn.Sequential(
            torch.nn.Conv1d(1, hidden_channels, kernel_size=25, stride=1, padding=12),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(kernel_size=3, stride=3),
            torch.nn.Dropout(dropout)
        )
        
        # Graph convolution layers for spatial learning
        self.graph_convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        conv_class = GCNConv if graph_conv == 'GCN' else GATConv
        
        for i in range(num_graph_layers):
            in_ch = hidden_channels
            
            if graph_conv == 'GAT':
                self.graph_convs.append(conv_class(in_ch, hidden_channels, heads=1))
            else:
                self.graph_convs.append(conv_class(in_ch, hidden_channels))
            
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))
        
        self.dropout = torch.nn.Dropout(dropout)
        
        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels // 2, num_classes)
        )
    
    def forward(self, data):
        """
        Args:
            data: PyG Data object with x, edge_index, edge_attr, batch
        
        Returns:
            logits: (batch_size, num_classes)
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # x shape: (total_nodes_in_batch, num_timepoints)
        
        # Temporal processing: (total_nodes, time_steps) -> (total_nodes, hidden)
        x = x.unsqueeze(1)  # (total_nodes, 1, time_steps)
        x = self.temporal_conv(x)  # (total_nodes, hidden, reduced_time)
        x = x.mean(dim=2)  # Average over time: (total_nodes, hidden)
        
        # Graph convolutions
        for conv, bn in zip(self.graph_convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling (pool all nodes from each graph)
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        
        x = global_mean_pool(x, batch)  # (batch_size, hidden)
        
        # Classification
        out = self.classifier(x)
        
        return out
    
class TimeThenSpaceModel(nn.Module):
    def __init__(self, input_size: int, n_nodes: int, horizon: int,
                 hidden_size: int = 32,
                 rnn_layers: int = 1,
                 gnn_kernel: int = 2):
        super(TimeThenSpaceModel, self).__init__()

        self.encoder = nn.Linear(input_size, hidden_size)

        self.node_embeddings = NodeEmbedding(n_nodes, hidden_size)

        self.time_nn = RNN(input_size=hidden_size,
                           hidden_size=hidden_size,
                           n_layers=rnn_layers,
                           cell='gru',
                           return_only_last_state=True)
        
        self.space_nn = DiffConv(in_channels=hidden_size,
                                 out_channels=hidden_size,
                                 k=gnn_kernel)

        self.decoder = nn.Linear(hidden_size, input_size * horizon)
        self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)

    def forward(self, x, edge_index, edge_weight):
        # x: [batch time nodes features]
        x_enc = self.encoder(x)  # linear encoder: x_enc = xΘ + b
        x_emb = x_enc + self.node_embeddings()  # add node-identifier embeddings
        h = self.time_nn(x_emb)  # temporal processing: x=[b t n f] -> h=[b n f]
        z = self.space_nn(h, edge_index, edge_weight)  # spatial processing
        x_out = self.decoder(z)  # linear decoder: z=[b n f] -> x_out=[b n t⋅f]
        x_horizon = self.rearrange(x_out)
        return x_horizon

# Training functions
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        loss = F.cross_entropy(out, data.y.squeeze())
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == data.y.squeeze()).sum().item()
        total += data.y.size(0)
    
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        correct += (pred == data.y.squeeze()).sum().item()
        total += data.y.size(0)
    
    return correct / total


if __name__ == "__main__":
    print("BCI Competition IV with PyTorch Geometric")
    print("=" * 60)
    
    # Set your data path here
    data_path = "/home/theniche/Research/Hilbert-Bundle-Convolutional-Networks/Data/BCICIV_2a_gdf"  # CHANGE THIS
    
    try:
        # Create dataset
        dataset = BCIGraphDataset(
            data_path=data_path,
            dataset_type='2a',
            subjects=['A01', 'A02', 'A03'],  # Use subset for quick test
            train=True,
            tmin=0.5,
            tmax=4.5,
            baseline=None,  # No baseline correction
            graph_type='distance',  # or 'fully_connected'
            k_neighbors=4,
            normalize=True
        )
        
        # Split train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create PyG DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Create model
        sample_data = dataset[0]
        in_channels = sample_data.x.shape[1]  # Number of time points
        
        model = EEG_GNN(
            in_channels=in_channels,
            hidden_channels=64,
            num_classes=4,
            num_graph_layers=3,
            dropout=0.5,
            graph_conv='GCN'  # or 'GAT'
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        
        print(f"\nModel architecture:")
        print(model)
        print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"\nDevice: {device}")
        
        # Training loop
        print("\nStarting training...")
        num_epochs = 1000
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
            val_acc = evaluate(model, val_loader, device)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:03d}: Loss: {train_loss:.4f}, '
                      f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
                      f'Best Val: {best_val_acc:.4f}')
        
        print(f"\nTraining complete!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo use this code:")
        print("1. Download BCI Competition IV dataset from:")
        print("   https://www.bbci.de/competition/iv/download/")
        print("2. Extract the GDF files")
        print("3. Update the 'data_path' variable above")
        print("4. Run again!")