"""
Generate PCA conceptual figures for lecture slides 8, 9, and 10
This script creates visualizations showing:
  1. First principal component
  2. Both principal components (orthogonal)
  3. Projection onto first principal component
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Apply custom matplotlib style if available
try:
    plt.style.use('custom.mplstyle')
    print("Using custom.mplstyle")
except OSError:
    print("Warning: custom.mplstyle not found, using default style")

# Set random seed for reproducibility
np.random.seed(42)

# Generate 2D elliptical data
n_points = 200
mean = [0, 0]
# Covariance matrix to create an elliptical distribution
# with major axis at ~45 degrees
cov = [[3.0, 1.5],
       [1.5, 1.0]]

# Generate the data
X = np.random.multivariate_normal(mean, cov, n_points)

# Perform PCA manually
# 1. Center the data
X_centered = X - np.mean(X, axis=0)

# 2. Compute covariance matrix
cov_matrix = np.cov(X_centered.T)

# 3. Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 4. Sort by eigenvalues (descending)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Principal components (unit vectors)
u1 = eigenvectors[:, 0]  # First principal component
u2 = eigenvectors[:, 1]  # Second principal component

# Scale arrows by square root of eigenvalues for visualization
scale1 = np.sqrt(eigenvalues[0]) * 2.0  # Scale factor for first PC
scale2 = np.sqrt(eigenvalues[1]) * 2.0  # Scale factor for second PC

# Common plotting parameters
plot_params = {
    'xlim': (-6, 6),
    'ylim': (-4, 4),
    'xlabel': '$x_1$',
    'ylabel': '$x_2$',
}


def plot_scatter(ax):
    """Plot the base scatter plot on given axes"""
    ax.scatter(X[:, 0], X[:, 1], c='#3498db', alpha=0.6,
               edgecolors='#2c3e50', label='Data points')
    ax.axhline(y=0, color='k', alpha=0.3)
    ax.axvline(x=0, color='k', alpha=0.3)
    ax.set_xlim(plot_params['xlim'])
    ax.set_ylim(plot_params['ylim'])
    ax.set_xlabel(plot_params['xlabel'])
    ax.set_ylabel(plot_params['ylabel'])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.2)


def add_arrow(ax, direction, scale, color, label):
    """Add a principal component arrow to the plot"""
    arrow = FancyArrowPatch((0, 0),
                           (direction[0] * scale, direction[1] * scale),
                           arrowstyle='->',
                           color=color,
                           zorder=5, label=label)
    ax.add_patch(arrow)

    # Add label near the tip
    tip_x = direction[0] * scale * 0.85
    tip_y = direction[1] * scale * 0.85
    ax.text(tip_x, tip_y, label, fontweight='bold',
            color=color, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor=color, alpha=0.8))


def generate_figure_0():
    """Generate Figure 0: Basic 2D Elliptical Data (Slide 6)"""
    print("Generating Figure 0: Basic 2D Elliptical Data...")

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_scatter(ax)
    ax.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()
    plt.savefig('../figures/slide06_elliptical_data.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: ../figures/slide06_elliptical_data.png")


def generate_figure_1():
    """Generate Figure 1: First Principal Component"""
    print("Generating Figure 1: First Principal Component...")

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_scatter(ax)
    add_arrow(ax, u1, scale1, '#e74c3c', r'$\mathbf{u}_1$')
    ax.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()
    plt.savefig('../figures/slide09_first_pc.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: ../figures/slide09_first_pc.png")


def generate_figure_2():
    """Generate Figure 2: Both Principal Components"""
    print("Generating Figure 2: Both Principal Components...")

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_scatter(ax)
    add_arrow(ax, u1, scale1, '#e74c3c', r'$\mathbf{u}_1$')
    add_arrow(ax, u2, scale2, '#27ae60', r'$\mathbf{u}_2$')
    ax.legend(loc='upper left', framealpha=0.9)

    # Add note about orthogonality
    ax.text(0.98, 0.02, 'Note: Components are orthogonal\n(perpendicular)',
            transform=ax.transAxes, verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('../figures/slide10_both_pcs.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: ../figures/slide10_both_pcs.png")


def generate_figure_3():
    """Generate Figure 3: Projection onto First Principal Component"""
    print("Generating Figure 3: Projection onto First Principal Component...")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Select a subset of points to show projections (to avoid clutter)
    n_projections = 30
    indices = np.linspace(0, n_points - 1, n_projections, dtype=int)

    # Plot all data points
    ax.scatter(X[:, 0], X[:, 1], c='#3498db', alpha=0.4,
               edgecolors='#2c3e50', label='Original points', zorder=2)

    # Draw the first principal component as a line (not arrow)
    line_extent = 8  # Extend line across the plot
    ax.plot([-u1[0] * line_extent, u1[0] * line_extent],
            [-u1[1] * line_extent, u1[1] * line_extent],
            'r-', label=r'$\mathbf{u}_1$ (first PC)', zorder=3)

    # For each selected point, show projection
    projections = []
    for i in indices:
        point = X[i]
        # Project point onto u1: projection = (point · u1) * u1
        projection_scalar = np.dot(point, u1)
        projection_point = projection_scalar * u1
        projections.append(projection_point)

        # Draw line from original point to projection
        ax.plot([point[0], projection_point[0]],
                [point[1], projection_point[1]],
                'gray', linestyle='--', alpha=1.0, zorder=1)

    # Plot projected points
    projections = np.array(projections)
    ax.scatter(projections[:, 0], projections[:, 1],
               c='#e74c3c', alpha=0.8, edgecolors='darkred',
               label='Projected points', zorder=4, marker='o')

    # Set up axes
    ax.axhline(y=0, color='k', alpha=0.3)
    ax.axvline(x=0, color='k', alpha=0.3)
    ax.set_xlim(plot_params['xlim'])
    ax.set_ylim(plot_params['ylim'])
    ax.set_xlabel(plot_params['xlabel'])
    ax.set_ylabel(plot_params['ylabel'])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.2)

    ax.legend(loc='lower right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig('../figures/slide11_projection.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: ../figures/slide11_projection.png")


def main():
    """Generate all PCA conceptual figures"""
    print("\n" + "=" * 60)
    print("Generating PCA Conceptual Figures for Slides 6, 9-11")
    print("=" * 60 + "\n")

    print(f"Data: {n_points} points from 2D elliptical distribution")
    print(f"Principal components computed via eigendecomposition")
    print(f"First PC explains {eigenvalues[0]/eigenvalues.sum()*100:.1f}% of variance")
    print(f"Second PC explains {eigenvalues[1]/eigenvalues.sum()*100:.1f}% of variance\n")

    # Generate all figures
    generate_figure_0()
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()

    print("\n" + "=" * 60)
    print("Successfully generated 4 figures")
    print("=" * 60)
    print("\nFiles created in ../figures/ directory:")
    print("  1. slide06_elliptical_data.png - Slide 6: Raw data cloud (300 DPI)")
    print("  2. slide09_first_pc.png        - Slide 9: First PC (300 DPI)")
    print("  3. slide10_both_pcs.png        - Slide 10: Both PCs (300 DPI)")
    print("  4. slide11_projection.png      - Slide 11: Projection (300 DPI)")
    print("\nThese files are ready to use in your LaTeX presentation.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
