"""
Generate PCA application figures for lecture slides
This script creates 4 visualization panels showing real-world PCA applications
All images will be saved in the current directory with standardized filenames
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# You'll need these packages installed:
# pip install numpy matplotlib scikit-learn

try:
    from sklearn.datasets import fetch_olivetti_faces, load_digits
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not installed. Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False


def generate_eigenfaces():
    """Generate eigenfaces visualization (Top-left panel)"""
    if not SKLEARN_AVAILABLE:
        return None

    print("Generating eigenfaces...")

    # Load Olivetti faces dataset (400 face images, 64x64 pixels)
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = faces.data

    # Perform PCA
    n_components = 12
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    pca.fit(X)

    # Create figure
    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    fig.suptitle('Eigenfaces (First 12 Principal Components)', fontsize=14, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        if i < n_components:
            ax.imshow(pca.components_[i].reshape(64, 64), cmap='gray')
            ax.set_title(f'PC{i+1}', fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('eigenfaces.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: eigenfaces.png")
    return True


def generate_genomics_pca():
    """Generate genomics PCA scatter plot (Top-right panel)"""
    print("Generating genomics PCA scatter plot...")

    # Simulate gene expression data for different tissue types
    np.random.seed(42)
    n_samples_per_group = 50
    n_genes = 1000

    # Create synthetic gene expression data for 3 tissue types
    # Each tissue type has different expression patterns
    tissue1 = np.random.randn(n_samples_per_group, n_genes) * 1.0 + np.array([2, -1] + [0]*(n_genes-2))
    tissue2 = np.random.randn(n_samples_per_group, n_genes) * 1.0 + np.array([-2, 1] + [0]*(n_genes-2))
    tissue3 = np.random.randn(n_samples_per_group, n_genes) * 1.0 + np.array([0, -2] + [0]*(n_genes-2))

    X = np.vstack([tissue1, tissue2, tissue3])
    labels = np.array(['Tissue A']*n_samples_per_group +
                      ['Tissue B']*n_samples_per_group +
                      ['Tissue C']*n_samples_per_group)

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'Tissue A': '#e74c3c', 'Tissue B': '#3498db', 'Tissue C': '#2ecc71'}

    for tissue_type in ['Tissue A', 'Tissue B', 'Tissue C']:
        mask = labels == tissue_type
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  c=colors[tissue_type], label=tissue_type,
                  s=50, alpha=0.6, edgecolors='k', linewidth=0.5)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title('Gene Expression PCA by Tissue Type', fontsize=14, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('genomics_pca.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: genomics_pca.png")
    return True


def generate_financial_pca():
    """Generate financial time series PCA (Bottom-left panel)"""
    print("Generating financial time series PCA...")

    # Simulate correlated stock prices
    np.random.seed(42)
    n_days = 252  # trading days in a year
    n_stocks = 5

    # Create correlated stock returns
    # Market factor + individual stock noise
    market_factor = np.cumsum(np.random.randn(n_days) * 0.01)

    stock_prices = np.zeros((n_days, n_stocks))
    stock_names = ['Stock A', 'Stock B', 'Stock C', 'Stock D', 'Stock E']

    for i in range(n_stocks):
        # Each stock follows market + some individual variation
        individual_noise = np.cumsum(np.random.randn(n_days) * 0.005)
        stock_prices[:, i] = 100 * np.exp(0.7 * market_factor + 0.3 * individual_noise)

    # Apply PCA to returns
    returns = np.diff(stock_prices, axis=0) / stock_prices[:-1]
    pca = PCA(n_components=2)
    returns_pca = pca.fit_transform(returns)

    # Create figure with subplots
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(1, 2, figure=fig)

    # Left: Original time series
    ax1 = fig.add_subplot(gs[0, 0])
    for i, name in enumerate(stock_names):
        ax1.plot(stock_prices[:, i], label=name, linewidth=1.5)
    ax1.set_xlabel('Trading Days', fontsize=10)
    ax1.set_ylabel('Price ($)', fontsize=10)
    ax1.set_title('Original Stock Prices', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Right: PCA projection
    ax2 = fig.add_subplot(gs[0, 1])
    scatter = ax2.scatter(returns_pca[:, 0], returns_pca[:, 1],
                         c=np.arange(len(returns_pca)), cmap='viridis',
                         s=20, alpha=0.6)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=10)
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=10)
    ax2.set_title('PCA of Stock Returns', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Time', fontsize=9)

    fig.suptitle('Financial Time Series PCA', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('financial_pca.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: financial_pca.png")
    return True


def generate_mnist_pca():
    """Generate MNIST PCA visualization (Bottom-right panel)"""
    if not SKLEARN_AVAILABLE:
        return None

    print("Generating MNIST PCA...")

    # Load MNIST digits
    digits = load_digits()
    X = digits.data

    # Perform PCA
    pca = PCA(n_components=64)
    X_pca = pca.fit_transform(X)

    # Create figure
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Show sample digit
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(digits.images[0], cmap='gray')
    ax1.set_title('Sample Digit (8×8)', fontsize=10, fontweight='bold')
    ax1.axis('off')

    # Show first 6 principal components as images
    for i in range(6):
        ax = fig.add_subplot(gs[i//3, i%3 + (1 if i < 3 else 0)])
        # Reshape PC to 8x8 image
        pc_image = pca.components_[i].reshape(8, 8)
        ax.imshow(pc_image, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
        variance_pct = pca.explained_variance_ratio_[i] * 100
        ax.set_title(f'PC{i+1} ({variance_pct:.1f}%)', fontsize=9)
        ax.axis('off')

    fig.suptitle('MNIST Digits: Principal Components', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mnist_pca.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: mnist_pca.png")
    return True


def main():
    """Generate all four PCA application figures"""
    print("\n" + "="*60)
    print("Generating PCA Application Figures")
    print("="*60 + "\n")

    success_count = 0

    # Generate each figure
    if generate_eigenfaces():
        success_count += 1

    if generate_genomics_pca():
        success_count += 1

    if generate_financial_pca():
        success_count += 1

    if generate_mnist_pca():
        success_count += 1

    print("\n" + "="*60)
    print(f"Successfully generated {success_count}/4 figures")
    print("="*60)
    print("\nFiles created:")
    print("  1. eigenfaces.png       - Eigenfaces for face recognition")
    print("  2. genomics_pca.png     - Gene expression PCA scatter plot")
    print("  3. financial_pca.png    - Financial time series analysis")
    print("  4. mnist_pca.png        - MNIST digit principal components")
    print("\nThese files can be directly used in your LaTeX presentation.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
