import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class KernelSimilarityOptimizer:
    """
    Finds optimal linear transformation T that maximizes k(TX, Y)
    where k is a kernel similarity measure.
    """
    
    def __init__(self, input_dim, output_dim, kernel_type='rbf', kernel_param=1.0):
        """
        Args:
            input_dim: Dimension of input time series X
            output_dim: Dimension of output time series Y
            kernel_type: 'rbf', 'linear', or 'polynomial'
            kernel_param: Kernel parameter (sigma for RBF, degree for polynomial)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_type = kernel_type
        self.kernel_param = kernel_param
        
        # Initialize transformation matrix T
        self.T = nn.Parameter(torch.randn(output_dim, input_dim) * 0.01)
        
    def compute_kernel(self, X, Y):
        """
        Compute kernel similarity k(X, Y)
        
        Args:
            X: (n_samples, dim) tensor
            Y: (n_samples, dim) tensor
        
        Returns:
            Scalar kernel similarity value
        """
        if self.kernel_type == 'rbf':
            # RBF kernel: exp(-||X - Y||^2 / (2 * sigma^2))
            # Mean over all pairwise similarities
            diff = X.unsqueeze(1) - Y.unsqueeze(0)  # (n, m, dim)
            sq_dist = torch.sum(diff ** 2, dim=2)  # (n, m)
            kernel_matrix = torch.exp(-sq_dist / (2 * self.kernel_param ** 2))
            return kernel_matrix.mean()
        
        elif self.kernel_type == 'linear':
            # Linear kernel: <X, Y> / (n * dim)
            return (X @ Y.T).mean()
        
        elif self.kernel_type == 'polynomial':
            # Polynomial kernel: (1 + <X, Y>)^d
            degree = int(self.kernel_param)
            return ((1 + X @ Y.T) ** degree).mean()
        
        elif self.kernel_type == 'cosine':
            # Cosine similarity
            X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)
            Y_norm = Y / (Y.norm(dim=1, keepdim=True) + 1e-8)
            return (X_norm @ Y_norm.T).mean()
        
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def optimize(self, X, Y, lr=0.01, n_iterations=1000, verbose=True):
        """
        Optimize T to maximize k(TX, Y)
        
        Args:
            X: (n_samples, input_dim) tensor - input time series
            Y: (n_samples, output_dim) tensor - target time series
            lr: Learning rate
            n_iterations: Number of gradient descent steps
            verbose: Whether to print progress
        
        Returns:
            T_optimal: Optimized transformation matrix
            history: Dictionary with training history
        """
        optimizer = optim.Adam([self.T], lr=lr)
        history = {'loss': [], 'kernel_sim': []}
        
        for i in range(n_iterations):
            optimizer.zero_grad()
            
            # Apply transformation
            TX = X @ self.T.T  # (n_samples, output_dim)
            
            # Compute kernel similarity
            kernel_sim = self.compute_kernel(TX, Y)
            
            # We want to maximize similarity, so minimize negative similarity
            loss = -kernel_sim
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Record history
            history['loss'].append(loss.item())
            history['kernel_sim'].append(kernel_sim.item())
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{n_iterations}, "
                      f"Kernel Similarity: {kernel_sim.item():.6f}")
        
        return self.T.detach(), history


def generate_example_data(n_samples=100, input_dim=10, output_dim=5, noise_level=0.1):
    """Generate synthetic time series data for testing"""
    # Generate random true transformation
    T_true = torch.randn(output_dim, input_dim)
    
    # Generate input data
    X = torch.randn(n_samples, input_dim)
    
    # Generate target data with some relationship to X
    Y = X @ T_true.T + noise_level * torch.randn(n_samples, output_dim)
    
    return X, Y, T_true


# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 200
    input_dim = 8
    output_dim = 5
    X, Y, T_true = generate_example_data(n_samples, input_dim, output_dim)
    
    print(f"Data shapes: X={X.shape}, Y={Y.shape}")
    print(f"True transformation T shape: {T_true.shape}\n")
    
    # Initialize optimizer with RBF kernel
    optimizer = KernelSimilarityOptimizer(
        input_dim=input_dim,
        output_dim=output_dim,
        kernel_type='rbf',
        kernel_param=1.0
    )
    
    # Optimize
    print("Optimizing transformation matrix T...")
    T_optimal, history = optimizer.optimize(X, Y, lr=0.01, n_iterations=1000)
    
    print(f"\nOptimization complete!")
    print(f"Final kernel similarity: {history['kernel_sim'][-1]:.6f}")
    
    # Plot training progress
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['kernel_sim'])
    plt.xlabel('Iteration')
    plt.ylabel('Kernel Similarity')
    plt.title('Kernel Similarity Over Training')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.xlabel('Iteration')
    plt.ylabel('Loss (Negative Similarity)')
    plt.title('Loss Over Training')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Compare with initial random transformation
    T_init = torch.randn(output_dim, input_dim) * 0.01
    TX_init = X @ T_init.T
    TX_optimal = X @ T_optimal.T
    
    init_sim = optimizer.compute_kernel(TX_init, Y)
    optimal_sim = optimizer.compute_kernel(TX_optimal, Y)
    
    print(f"\nInitial random T kernel similarity: {init_sim.item():.6f}")
    print(f"Optimized T kernel similarity: {optimal_sim.item():.6f}")
    print(f"Improvement: {(optimal_sim - init_sim).item():.6f}")
