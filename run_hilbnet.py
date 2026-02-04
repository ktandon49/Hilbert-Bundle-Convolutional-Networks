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
from Repo.Parallel_Transport.HilbNetArtchitecture import HilbNet



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
        pred = F.softmax(out, dim=1).argmax(dim=1)
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
        pred = F.softmax(out, dim=1).argmax(dim=1)
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
            f"{model_name}/best_val_acc": best_val_acc,
        })
    
    print(f"\n{model_name} - Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    return best_val_acc


# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Hilbnet for EEG classification')

    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Hidden layer size (default: 64)')
    parser.add_argument('--rnn_layers', type=int, default=1,
                        help='Number of RNN layers (default: 1)')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of RNN layers (default: 2)')
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
    parser.add_argument('--poly_dim', type=int, default=3,
                        help='Highest degree of the polynomial (default:3)')
    
    args = parser.parse_args()
    
    
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
    print(f"Models to train: HilbNet")
    print(f"Subject: {args.subject}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"RNN layers: {args.rnn_layers}")
    print(f"GNN kernel: {args.gnn_kernel}")
    print(f"Hilbnet layers: {args.n_layers}")
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
            "models": 'HilbNet',
            'poly_dim': args.poly_dim
        }
    )
    
    results = {}
    

    hilbnet_model = HilbNet(
        input_size=1,
        n_nodes=sample.num_nodes,
        time_steps=dataset.time_steps,
        hidden_size=args.hidden_size,
        n_classes=dataset.n_classes,
        n_layers=2,
        pt_maps=dataset.pt_maps.to(device),
        poly_dim=args.poly_dim
    )
    results['HilbNet'] = train_model(
        hilbnet_model, "HilbNet", train_loader, val_loader, device, 
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
