"""
Generate figure showing PCA limitations for slide 45
This script creates a 2-panel figure demonstrating:
  1. Swiss roll: Nonlinear manifold where PCA fails
  2. Low-variance discrimination: When discriminative info is in low-variance directions
"""

import numpy as np
import matplotlib.pyplot as plt

# Apply custom matplotlib style if available
try:
    plt.style.use('custom.mplstyle')
    print("Using custom.mplstyle")
except OSError:
    print("Warning: custom.mplstyle not found, using default style")

# Set random seed for reproducibility
np.random.seed(42)

def generate_swiss_roll(n_samples=1000, noise=0.1):
    """Generate Swiss roll data and return 2D projection"""
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    x = t * np.cos(t)
    y = 21 * np.random.rand(n_samples)
    z = t * np.sin(t)

    # Add noise
    x += noise * np.random.randn(n_samples)
    y += noise * np.random.randn(n_samples)
    z += noise * np.random.randn(n_samples)

    # Return 2D projection (x-z plane shows the roll structure)
    return x, z, t


def generate_parallel_clusters(n_samples=200):
    """Generate two elongated parallel clusters"""
    # Cluster 1: centered at y=1, elongated along x
    x1 = np.random.randn(n_samples // 2) * 3  # Large variance in x
    y1 = np.random.randn(n_samples // 2) * 0.3 + 1  # Small variance in y, centered at 1

    # Cluster 2: centered at y=-1, elongated along x
    x2 = np.random.randn(n_samples // 2) * 3  # Large variance in x
    y2 = np.random.randn(n_samples // 2) * 0.3 - 1  # Small variance in y, centered at -1

    return x1, y1, x2, y2


def main():
    """Generate the 2-panel PCA limitations figure"""
    print("\n" + "=" * 60)
    print("Generating PCA Limitations Figure")
    print("=" * 60 + "\n")

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- LEFT PANEL: Swiss Roll (2D) ---
    print("Generating Swiss roll (2D projection)...")

    x, z, t = generate_swiss_roll(n_samples=800, noise=0.5)

    # Color by position along the roll
    scatter = ax1.scatter(x, z, c=t, cmap='viridis', s=20, alpha=0.6, edgecolors='k', linewidth=0.3)

    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Z', fontsize=12)
    ax1.set_title('Swiss Roll: Nonlinear Manifold (2D Projection)', fontsize=13, fontweight='bold', pad=15)
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Position along roll', fontsize=10)

    # Add text box explaining the problem
    textstr = 'PCA captures linear trends\nbut misses rolled structure'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    # --- RIGHT PANEL: Low-Variance Discrimination ---
    print("Generating low-variance classification example...")

    x1, y1, x2, y2 = generate_parallel_clusters(n_samples=300)

    # Plot the two clusters
    ax2.scatter(x1, y1, c='#e74c3c', s=40, alpha=0.6, edgecolors='black', linewidth=0.5, label='Class 1')
    ax2.scatter(x2, y2, c='#3498db', s=40, alpha=0.6, edgecolors='black', linewidth=0.5, label='Class 2')

    # Compute and show PCA directions
    # Combine data
    X_combined = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    X_centered = X_combined - X_combined.mean(axis=0)

    # Covariance matrix
    cov = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Plot PC directions from origin
    origin = X_combined.mean(axis=0)
    scale_pc1 = 4  # PC1 gets longer arrow
    scale_pc2 = 2  # PC2 gets shorter arrow (half the length)

    # PC1 (high variance, useless for classification)
    ax2.arrow(origin[0], origin[1],
             eigenvectors[0, 0] * scale_pc1, eigenvectors[1, 0] * scale_pc1,
             head_width=0.3, head_length=0.3, fc='green', ec='green',
             linewidth=2.5, alpha=0.7, label='PC1 (high var.)')

    # PC2 (low variance, discriminative) - SHORTER arrow
    ax2.arrow(origin[0], origin[1],
             eigenvectors[0, 1] * scale_pc2, eigenvectors[1, 1] * scale_pc2,
             head_width=0.3, head_length=0.3, fc='orange', ec='orange',
             linewidth=2.5, alpha=0.7, label='PC2 (low var., discriminative)')

    ax2.set_xlabel('Feature 1', fontsize=12)
    ax2.set_ylabel('Feature 2', fontsize=12)
    ax2.set_title('Discrimination in Low-Variance Direction', fontsize=13, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    ax2.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax2.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

    # Add text box - TOP LEFT corner
    variance_ratio = eigenvalues / eigenvalues.sum()
    textstr = f'PC1: {variance_ratio[0]*100:.1f}% variance\nPC2: {variance_ratio[1]*100:.1f}% variance\n(But PC2 separates classes!)'
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    # Save figure
    output_path = '../figures/slide45_pca_failures.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Reset style
    plt.style.use('default')

    print(f"\nSaved: {output_path}")
    print("\n" + "=" * 60)
    print("PCA Limitations Figure Complete")
    print("=" * 60 + "\n")

    print(f"Variance breakdown for low-variance example:")
    print(f"  PC1: {variance_ratio[0]*100:.1f}% (along clusters)")
    print(f"  PC2: {variance_ratio[1]*100:.1f}% (between clusters)")


if __name__ == "__main__":
    main()
