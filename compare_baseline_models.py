import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import unbatch
from tsl.nn.blocks.encoders import RNN
from tsl.nn.layers import NodeEmbedding, DiffConv
from torch_geometric.nn import GCNConv, global_mean_pool
from einops import rearrange
import wandb
import argparse


class GNNModel(nn.Module):
    """
    Spatial-only model using GNN.
    Processes each timestep independently, then pools over time.
    """
    def __init__(self, input_size: int, n_nodes: int, 
                 hidden_size: int = 32,
                 gnn_kernel: int = 2,
                 n_classes: int = 4):
        super().__init__()
        
        self.n_nodes = n_nodes
        self.encoder = nn.Linear(input_size, hidden_size)
        self.node_embeddings = NodeEmbedding(n_nodes, hidden_size)
        
        self.convs = nn.ModuleList()
        for _ in range(gnn_kernel):
            self.convs.append(GCNConv(hidden_size, hidden_size))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, n_classes)
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        
        batch_size = batch.max().item() + 1
        time_steps = x.shape[1]
        
        # Reshape
        x = x.view(batch_size, self.n_nodes, time_steps)
        x = rearrange(x, 'b n t -> b t n 1')
        
        # Encode
        x_enc = self.encoder(x)
        x_emb = x_enc + self.node_embeddings()
        
        # KEY CHANGE: Flatten batch and time dimensions together
        # [b, t, n, f] -> [(b*t), n, f] -> [(b*t)*n, f]
        x_flat = rearrange(x_emb, 'b t n f -> (b t n) f')
        
        # Create expanded batch indices for all (batch, timestep) pairs
        # Each node in each timestep gets the same graph ID
        batch_expanded = torch.repeat_interleave(
            torch.arange(batch_size * time_steps, device=batch.device),
            self.n_nodes
        )
        
        # Now apply ALL GNN layers at once across all timesteps
        h = x_flat
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
        
        # Pool each (batch, timestep) graph
        graph_embeddings = global_mean_pool(h, batch_expanded)  # [(b*t), hidden]
        
        # Reshape back and average over time
        graph_embeddings = graph_embeddings.view(batch_size, time_steps, -1)
        out = graph_embeddings.mean(dim=1)  # [b, hidden]
        
        out = self.classifier(out)
        return out


class RNNModel(nn.Module):
    """
    Temporal-only model using RNN.
    Processes temporal dynamics for each node independently, then pools.
    """
    def __init__(self, input_size: int, n_nodes: int, 
                 hidden_size: int = 32,
                 rnn_layers: int = 1,
                 n_classes: int = 4):
        super().__init__()
        
        self.n_nodes = n_nodes
        self.encoder = nn.Linear(input_size, hidden_size)
        self.node_embeddings = NodeEmbedding(n_nodes, hidden_size)
        
        self.time_nn = RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            n_layers=rnn_layers,
            cell='gru',
            return_only_last_state=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, n_classes)
        )

    def forward(self, data):
        x = data.x
        batch = data.batch
        
        batch_size = batch.max().item() + 1
        time_steps = x.shape[1]
        
        # Reshape: [total_nodes, time] -> [batch, nodes, time] -> [batch, time, nodes, 1]
        x = x.view(batch_size, self.n_nodes, time_steps)
        x = rearrange(x, 'b n t -> b t n 1')
        
        # Encode
        x_enc = self.encoder(x)                    # [b, t, n, hidden]
        x_emb = x_enc + self.node_embeddings()
        
        # Temporal processing: [b, t, n, f] -> [b, n, f]
        h = self.time_nn(x_emb)
        
        # Pool over nodes
        out = h.mean(dim=1)  # [b, hidden]
        out = self.classifier(out)
        return out


class TimeThenSpaceGNNModel(nn.Module):
    """
    Combined temporal-then-spatial model.
    First applies RNN to capture temporal dynamics, then GNN for spatial relationships.
    """
    def __init__(self, input_size: int, n_nodes: int, 
                 hidden_size: int = 32,
                 rnn_layers: int = 1,
                 gnn_kernel: int = 2,
                 n_classes: int = 4):
        super().__init__()
        
        self.n_nodes = n_nodes
        self.encoder = nn.Linear(input_size, hidden_size)
        self.node_embeddings = NodeEmbedding(n_nodes, hidden_size)
        
        self.time_nn = RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            n_layers=rnn_layers,
            cell='gru',
            return_only_last_state=True
        )
        
        self.space_nn = DiffConv(
            in_channels=hidden_size,
            out_channels=hidden_size,
            k=gnn_kernel
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, n_classes)
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_weight = data.edge_attr.flatten()
        batch = data.batch
        
        batch_size = batch.max().item() + 1
        time_steps = x.shape[1]
        
        # Reshape: [total_nodes, time] -> [batch, nodes, time] -> [batch, time, nodes, 1]
        x = x.view(batch_size, self.n_nodes, time_steps)
        x = rearrange(x, 'b n t -> b t n 1')
        
        # Encode
        x_enc = self.encoder(x)                    # [b, t, n, hidden]
        x_emb = x_enc + self.node_embeddings()
        
        # Temporal processing: [b, t, n, f] -> [b, n, f]
        h = self.time_nn(x_emb)
        
        # Spatial processing
        h_flat = rearrange(h, 'b n f -> (b n) f')
        z_flat = self.space_nn(h_flat, edge_index, edge_weight)
        
        # Global pooling
        out = global_mean_pool(z_flat, batch)
        out = self.classifier(out)
        return out


def train_epoch(model, loader, optimizer, device):
    """Training loop for one epoch"""
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
    """Evaluation loop"""
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


def train_model(model, model_name, train_loader, val_loader, device, 
                epochs=1000, lr=0.001, weight_decay=5e-4):
    """Train a single model and track results"""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_acc = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}: Loss={train_loss:.4f}, Train={train_acc:.4f}, Val={val_acc:.4f}')
        
        wandb.log({
            f"{model_name}/epoch": epoch,
            f"{model_name}/train_loss": train_loss,
            f"{model_name}/train_acc": train_acc,
            f"{model_name}/val_acc": val_acc,
        })
    
    print(f"\n{model_name} - Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    return best_val_acc


# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and compare EEG classification models')
    parser.add_argument('--models', type=str, nargs='+', 
                        choices=['gnn', 'rnn', 'timespace', 'all'],
                        default=['all'],
                        help='Which models to train: gnn, rnn, timespace, or all (default: all)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Hidden layer size (default: 64)')
    parser.add_argument('--rnn_layers', type=int, default=1,
                        help='Number of RNN layers (default: 1)')
    parser.add_argument('--gnn_kernel', type=int, default=2,
                        help='GNN kernel size/number of layers (default: 2)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (default: 5e-4)')
    parser.add_argument('--subject', type=str, default='A01',
                        help='Subject ID (default: A01)')
    parser.add_argument('--wandb_project', type=str, default='eeg-model-comparison',
                        help='Wandb project name (default: eeg-model-comparison)')
    
    args = parser.parse_args()
    
    # Determine which models to train
    if 'all' in args.models:
        models_to_train = ['gnn', 'rnn', 'spacetime']
    else:
        models_to_train = args.models
    
    import Journal_repo.data_util as data_util
    
    # Data loading
    data_path = "/home/theniche/Research/Hilbert-Bundle-Convolutional-Networks/Data/BCICIV_2a_gdf"
    dataset = data_util.BCIGraphDataset(
        data_path=data_path,
        dataset_type='2a',
        subjects=[args.subject],
        train=True,
        graph_type='fully_connected',
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    sample = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"{'='*60}")
    print(f"Models to train: {', '.join(models_to_train)}")
    print(f"Subject: {args.subject}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"RNN layers: {args.rnn_layers}")
    print(f"GNN kernel: {args.gnn_kernel}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Device: {device}")
    print(f"Dataset size: {len(dataset)} (train: {train_size}, val: {val_size})")
    print(f"{'='*60}\n")
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        config={
            "dataset": "BCICIV_2a",
            "subject": args.subject,
            "epochs": args.epochs,
            "hidden_size": args.hidden_size,
            "rnn_layers": args.rnn_layers,
            "gnn_kernel": args.gnn_kernel,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "models": models_to_train
        }
    )
    
    results = {}
    
    # 1. Train GNN model (if selected)
    if 'gnn' in models_to_train:
        gnn_model = GNNModel(
            input_size=1,
            n_nodes=sample.num_nodes,
            hidden_size=args.hidden_size,
            gnn_kernel=args.gnn_kernel,
            n_classes=dataset.n_classes
        )
        results['GNN'] = train_model(
            gnn_model, "GNN", train_loader, val_loader, device, 
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay
        )
    
    # 2. Train RNN model (if selected)
    if 'rnn' in models_to_train:
        rnn_model = RNNModel(
            input_size=1,
            n_nodes=sample.num_nodes,
            hidden_size=args.hidden_size,
            rnn_layers=args.rnn_layers,
            n_classes=dataset.n_classes
        )
        results['RNN'] = train_model(
            rnn_model, "RNN", train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay
        )
    
    # 3. Train SpaceTimeGNN model (if selected)
    if 'timespace' in models_to_train:
        spacetime_model = TimeThenSpaceGNNModel(
            input_size=1,
            n_nodes=sample.num_nodes,
            hidden_size=args.hidden_size,
            rnn_layers=args.rnn_layers,
            gnn_kernel=args.gnn_kernel,
            n_classes=dataset.n_classes
        )
        results['SpaceTimeGNN'] = train_model(
            spacetime_model, "SpaceTimeGNN", train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay
        )
    
    # Print final comparison
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("FINAL RESULTS COMPARISON")
        print(f"{'='*60}")
        for model_name, acc in results.items():
            print(f"{model_name:15s}: {acc:.4f}")
        
        best_model = max(results.items(), key=lambda x: x[1])
        print(f"\nBest model: {best_model[0]} with accuracy {best_model[1]:.4f}")
    elif len(results) == 1:
        model_name, acc = list(results.items())[0]
        print(f"\n{model_name} final accuracy: {acc:.4f}")
    
    wandb.finish()
