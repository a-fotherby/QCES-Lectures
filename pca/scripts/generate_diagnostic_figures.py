"""
Generate PCA diagnostic figures from first principles
This script:
  1. Generates synthetic high-dimensional data
  2. Performs PCA via eigendecomposition
  3. Creates diagnostic plots using actual eigenvalues
All images saved in ../figures/ directory
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

# Generate synthetic data with controlled eigenvalue structure
print("Generating synthetic dataset...")

n_samples = 100
n_features = 20

# Design eigenvalue spectrum with fast exponential decay
# This creates a clear "elbow" pattern with faster decay rate
eigenvalues_true = np.array([10 * (0.45 ** i) + 0.05 for i in range(n_features)])

print(f"\nDesigned eigenvalue spectrum:")
print(f"  λ₁ = {eigenvalues_true[0]:.2f} (largest)")
print(f"  λ₅ = {eigenvalues_true[4]:.2f} (near elbow)")
print(f"  λ₂₀ = {eigenvalues_true[19]:.2f} (noise floor)")

# Generate random orthogonal eigenvector matrix
# Using QR decomposition of random matrix
random_matrix = np.random.randn(n_features, n_features)
eigenvectors_true, _ = np.linalg.qr(random_matrix)

# Construct covariance matrix: C = Q @ Λ @ Q.T
covariance_matrix = eigenvectors_true @ np.diag(eigenvalues_true) @ eigenvectors_true.T

# Generate synthetic data from multivariate normal distribution
X = np.random.multivariate_normal(np.zeros(n_features), covariance_matrix, n_samples)

print(f"\nGenerated data shape: {X.shape}")
print(f"  {n_samples} samples × {n_features} features")

# Perform PCA from first principles
print("\nPerforming PCA from first principles...")

# Step 1: Center the data
X_centered = X - np.mean(X, axis=0)

# Step 2: Compute sample covariance matrix
S = (X_centered.T @ X_centered) / (n_samples - 1)

# Step 3: Eigendecomposition
eigenvalues_computed, eigenvectors_computed = np.linalg.eig(S)

# Step 4: Sort by eigenvalue magnitude (descending)
idx = np.argsort(eigenvalues_computed)[::-1]
eigenvalues_sorted = eigenvalues_computed[idx]
eigenvectors_sorted = eigenvectors_computed[:, idx]

# Ensure eigenvalues are real (should be, since S is symmetric)
eigenvalues_sorted = np.real(eigenvalues_sorted)

print(f"\nPCA Results:")
print(f"  Computed eigenvalues: {eigenvalues_sorted[:5]}")
print(f"  Sum of eigenvalues: {eigenvalues_sorted.sum():.2f}")

# Calculate variance explained
total_variance = eigenvalues_sorted.sum()
variance_explained_ratio = eigenvalues_sorted / total_variance
cumulative_variance_explained = np.cumsum(variance_explained_ratio)

print(f"\nVariance explained by first 5 components:")
for i in range(5):
    print(f"  PC{i+1}: {variance_explained_ratio[i]*100:.1f}%")

print(f"\nCumulative variance:")
print(f"  First 5 components: {cumulative_variance_explained[4]*100:.1f}%")
print(f"  First 10 components: {cumulative_variance_explained[9]*100:.1f}%")


def generate_scree_plot():
    """Generate scree plot showing eigenvalue decay"""
    print("\nGenerating scree plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    component_numbers = np.arange(1, n_features + 1)

    # Plot eigenvalues with log scale on y-axis
    ax.plot(component_numbers, eigenvalues_sorted, 'o-',
            color='#3498db', linewidth=2, markersize=8,
            markerfacecolor='#3498db', markeredgecolor='#2c3e50',
            markeredgewidth=1.5, label='Eigenvalues')

    # Set log scale for y-axis to better show the decay
    ax.set_yscale('log')

    # Find elbow point (where curvature changes)
    # Set elbow position based on visual inspection of the decay pattern
    elbow_component = 9
    elbow_idx = elbow_component - 1

    # Mark the elbow
    ax.axvline(x=elbow_component, color='gray', linestyle='--',
               linewidth=2, alpha=0.7, label=f'Elbow (M ≈ {elbow_component})')

    # Add annotation for elbow
    ax.annotate(f'Elbow suggests M ≈ {elbow_component}',
                xy=(elbow_component, eigenvalues_sorted[elbow_idx]),
                xytext=(elbow_component + 3, eigenvalues_sorted[elbow_idx] * 2),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', lw=2))

    # Add noise floor line (last few eigenvalues)
    noise_floor = np.mean(eigenvalues_sorted[-5:])
    ax.axhline(y=noise_floor, color='#e74c3c', linestyle='--',
               linewidth=2, alpha=0.7, label=f'Noise floor ≈ {noise_floor:.2f}')

    # Labels and title
    ax.set_xlabel('Principal Component', fontsize=13)
    ax.set_ylabel('Eigenvalue $\\lambda_i$', fontsize=13)
    ax.set_title('Scree Plot: Eigenvalue Decay', fontsize=14, fontweight='bold', pad=15)

    # Set x-axis limits and ticks
    ax.set_xlim(0.5, n_features + 0.5)
    ax.set_xticks(range(1, n_features + 1, 2))

    # Grid
    ax.grid(True, alpha=0.3, which='both', linestyle=':')

    # Legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plt.savefig('../figures/slide34_scree_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: ../figures/slide34_scree_plot.png")


def generate_cumulative_variance_plot():
    """Generate cumulative variance explained plot"""
    print("Generating cumulative variance plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    component_numbers = np.arange(1, n_features + 1)

    # Plot cumulative variance as percentage
    ax.plot(component_numbers, cumulative_variance_explained * 100,
            '-', color='#3498db', linewidth=3, label='Cumulative Variance')

    # Add threshold lines and markers
    thresholds = [90, 95, 99]
    colors = ['#27ae60', '#f39c12', '#e74c3c']

    for threshold, color in zip(thresholds, colors):
        # Horizontal line at threshold
        ax.axhline(y=threshold, color=color, linestyle='--',
                   linewidth=2, alpha=0.6, label=f'{threshold}% threshold')

        # Find number of components needed
        n_components_needed = np.where(cumulative_variance_explained * 100 >= threshold)[0][0] + 1

        # Vertical line showing components needed
        ax.axvline(x=n_components_needed, color=color, linestyle=':',
                   linewidth=2, alpha=0.6)

        # Annotation
        ax.annotate(f'M = {n_components_needed} for {threshold}%',
                    xy=(n_components_needed, threshold),
                    xytext=(n_components_needed + 1, threshold - 5),
                    fontsize=10, fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor=color, alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # Labels and title
    ax.set_xlabel('Number of Components M', fontsize=13)
    ax.set_ylabel('Cumulative Variance Explained (%)', fontsize=13)
    ax.set_title('Cumulative Variance Explained by Principal Components',
                 fontsize=14, fontweight='bold', pad=15)

    # Set axis limits
    ax.set_xlim(0.5, n_features + 0.5)
    ax.set_ylim(0, 105)

    # Set x-axis ticks
    ax.set_xticks(range(1, n_features + 1, 2))

    # Grid
    ax.grid(True, alpha=0.3, linestyle=':')

    # Legend
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig('../figures/slide35_cumulative_variance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: ../figures/slide35_cumulative_variance.png")


def main():
    """Generate all diagnostic PCA figures"""
    print("\n" + "=" * 60)
    print("Generating PCA Diagnostic Figures from First Principles")
    print("=" * 60 + "\n")

    generate_scree_plot()
    generate_cumulative_variance_plot()

    print("\n" + "=" * 60)
    print("Successfully generated 2 diagnostic figures")
    print("=" * 60)
    print("\nFiles created in ../figures/ directory:")
    print("  1. slide34_scree_plot.png            - Slide 34: Scree Plot (300 DPI)")
    print("  2. slide35_cumulative_variance.png   - Slide 35: Cumulative Variance (300 DPI)")
    print("\nThese files are ready to use in your LaTeX presentation.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
