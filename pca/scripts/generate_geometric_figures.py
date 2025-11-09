"""
Generate geometric PCA figures for lecture
This script creates 3D geometric visualizations showing:
  1. Projection operation and reconstruction error
All images will be saved in ../figures/ directory
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Apply custom matplotlib style if available
# Note: custom.mplstyle may have issues with 3D plots, so we skip it
# and use default styling for 3D geometric figures
print("Using default matplotlib style for 3D visualization")


def generate_3d_projection_figure():
    """Generate 3D projection and reconstruction error visualization"""
    print("Generating 3D Projection Figure...")

    # Create figure with 3D axes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define the original point in 3D space
    original_point = np.array([2.5, 1.5, 2.8])

    # Define the 2D subspace as a plane through the origin
    # We'll use a plane defined by two basis vectors (representing first 2 PCs)
    # For visualization, let's use a tilted plane
    u1 = np.array([1, 0, 0.3])  # First basis vector (normalized later)
    u2 = np.array([0, 1, 0.2])  # Second basis vector (normalized later)

    # Normalize basis vectors
    u1 = u1 / np.linalg.norm(u1)
    u2 = u2 / np.linalg.norm(u2)

    # Make u2 orthogonal to u1 using Gram-Schmidt
    u2 = u2 - np.dot(u2, u1) * u1
    u2 = u2 / np.linalg.norm(u2)

    # Calculate the projection of the original point onto the 2D subspace
    projection_coord1 = np.dot(original_point, u1)
    projection_coord2 = np.dot(original_point, u2)
    projected_point = projection_coord1 * u1 + projection_coord2 * u2

    # Draw the 2D subspace plane
    plane_range = 4
    xx, yy = np.meshgrid(np.linspace(-plane_range, plane_range, 10),
                         np.linspace(-plane_range, plane_range, 10))

    # Plane equation: any point on plane is c1*u1 + c2*u2
    # We need to express this in terms of xx, yy grid
    # For visualization, we'll create a plane surface
    plane_points = []
    for i in range(len(xx)):
        for j in range(len(yy[0])):
            c1, c2 = xx[i, j], yy[i, j]
            point = c1 * u1 + c2 * u2
            plane_points.append(point)

    plane_points = np.array(plane_points)

    # Create mesh for plane surface
    # Simpler approach: create a rectangular mesh on the plane
    corners = [
        -plane_range * u1 - plane_range * u2,
        plane_range * u1 - plane_range * u2,
        plane_range * u1 + plane_range * u2,
        -plane_range * u1 + plane_range * u2
    ]

    # Draw plane as a polygon
    verts = [corners]
    plane = Poly3DCollection(verts, alpha=0.3, facecolor='lightgray',
                            edgecolor='gray', linewidth=1)
    ax.add_collection3d(plane)

    # Plot the original point
    ax.scatter(*original_point, c='#3498db', s=200,
               edgecolors='#2c3e50', linewidth=2,
               label=r'Original $\tilde{\mathbf{x}}_n$',
               zorder=5, depthshade=False)

    # Plot the projected point
    ax.scatter(*projected_point, c='#e74c3c', s=200,
               edgecolors='darkred', linewidth=2,
               label=r'Reconstructed $\hat{\mathbf{x}}_n$',
               zorder=5, depthshade=False)

    # Draw the perpendicular connection (reconstruction error)
    ax.plot([original_point[0], projected_point[0]],
            [original_point[1], projected_point[1]],
            [original_point[2], projected_point[2]],
            'gray', linestyle='--', linewidth=2, alpha=0.8, zorder=4)

    # Add label for reconstruction error
    mid_point = (original_point + projected_point) / 2
    ax.text(mid_point[0], mid_point[1], mid_point[2],
            'Reconstruction\nerror', fontsize=10, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     alpha=0.7, edgecolor='gray'))

    # Draw basis vectors on the plane to indicate the subspace
    arrow_scale = 1.5
    ax.quiver(0, 0, 0, u1[0]*arrow_scale, u1[1]*arrow_scale, u1[2]*arrow_scale,
              color='#27ae60', arrow_length_ratio=0.15, linewidth=2, alpha=0.7)
    ax.quiver(0, 0, 0, u2[0]*arrow_scale, u2[1]*arrow_scale, u2[2]*arrow_scale,
              color='#27ae60', arrow_length_ratio=0.15, linewidth=2, alpha=0.7)

    # Label for 2D subspace
    ax.text(0, 0, -0.5, '2D Subspace\n(span of first 2 PCs)',
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray',
                     alpha=0.7, edgecolor='gray'))

    # Set labels and limits
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')

    # Set equal aspect ratio and limits
    max_range = 3.5
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, max_range])

    # Set viewing angle for best perspective
    ax.view_init(elev=20, azim=45)

    # Add legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/slide25_projection_3d.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: ../figures/slide25_projection_3d.png")


def generate_orthogonal_decomposition_figure():
    """Generate 2D orthogonal decomposition showing variance-error equivalence"""
    print("Generating Orthogonal Decomposition Figure...")

    # Create figure with two side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Define the original data point (as a vector from origin)
    original_vector = np.array([3.0, 2.0])

    # Define the 1D subspace direction (first principal component)
    subspace_direction = np.array([1.0, 0.3])
    subspace_direction = subspace_direction / np.linalg.norm(subspace_direction)

    # Calculate parallel (projection) and perpendicular components
    parallel_magnitude = np.dot(original_vector, subspace_direction)
    parallel_component = parallel_magnitude * subspace_direction
    perpendicular_component = original_vector - parallel_component

    # Function to draw the decomposition on an axis
    def draw_decomposition(ax, show_variance_labels=False):
        # Set limits and aspect
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 2.5)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.8)
        ax.axvline(x=0, color='k', linewidth=0.8)

        # Draw the 1D subspace line
        line_extent = 4
        ax.plot([-line_extent * subspace_direction[0], line_extent * subspace_direction[0]],
                [-line_extent * subspace_direction[1], line_extent * subspace_direction[1]],
                'gray', linewidth=2, alpha=0.5, linestyle='--',
                label='1D Subspace (span of $\\mathbf{u}_1$)')

        # Draw original vector
        ax.annotate('', xy=original_vector, xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color='#3498db',
                                 lw=3, mutation_scale=20))
        ax.text(original_vector[0] + 0.1, original_vector[1] + 0.1,
               r'$\tilde{\mathbf{x}}_n$', fontsize=14, fontweight='bold',
               color='#3498db')

        # Draw parallel component (projection)
        ax.annotate('', xy=parallel_component, xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color='#27ae60',
                                 lw=2.5, mutation_scale=20))

        # Draw perpendicular component (reconstruction error)
        ax.annotate('', xy=original_vector, xytext=parallel_component,
                   arrowprops=dict(arrowstyle='->', color='#e74c3c',
                                 lw=2.5, mutation_scale=20))

        # Add labels based on panel type
        if not show_variance_labels:
            # Left panel: geometric labels
            ax.text(parallel_component[0]/2 - 0.3, parallel_component[1]/2 - 0.2,
                   'Parallel\ncomponent', fontsize=11,
                   color='#27ae60', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           alpha=0.8, edgecolor='#27ae60'))

            ax.text((parallel_component[0] + original_vector[0])/2 + 0.2,
                   (parallel_component[1] + original_vector[1])/2,
                   'Perpendicular\ncomponent', fontsize=11,
                   color='#e74c3c', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           alpha=0.8, edgecolor='#e74c3c'))

            # Draw right angle symbol
            corner_size = 0.15
            corner = parallel_component
            v1 = perpendicular_component / np.linalg.norm(perpendicular_component) * corner_size
            v2 = subspace_direction * corner_size
            corner_points = np.array([
                corner + v2,
                corner + v2 + v1,
                corner + v1
            ])
            ax.plot([corner_points[0, 0], corner_points[1, 0], corner_points[2, 0]],
                   [corner_points[0, 1], corner_points[1, 1], corner_points[2, 1]],
                   'k-', linewidth=1.5)

        else:
            # Right panel: variance interpretation labels
            ax.text(parallel_component[0]/2 - 0.3, parallel_component[1]/2 - 0.3,
                   'Captured\n(large variance)', fontsize=11,
                   color='#27ae60', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           alpha=0.8, edgecolor='#27ae60'))

            ax.text((parallel_component[0] + original_vector[0])/2 + 0.2,
                   (parallel_component[1] + original_vector[1])/2,
                   'Lost\n(small variance)', fontsize=11,
                   color='#e74c3c', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           alpha=0.8, edgecolor='#e74c3c'))

            # Add equivalence note
            ax.text(0.5, 0.05, 'Minimizing red ↔ Maximizing green',
                   transform=ax.transAxes, fontsize=10,
                   ha='center', style='italic',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                           alpha=0.8, edgecolor='gray'))

        ax.set_xlabel('$x_1$', fontsize=12)
        ax.set_ylabel('$x_2$', fontsize=12)

    # Draw left panel: Geometric decomposition
    draw_decomposition(ax1, show_variance_labels=False)
    ax1.set_title('Geometric Decomposition', fontsize=14, fontweight='bold', pad=10)

    # Draw right panel: Variance interpretation
    draw_decomposition(ax2, show_variance_labels=True)
    ax2.set_title('Variance Interpretation', fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig('../figures/slide30_orthogonal_decomp.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: ../figures/slide30_orthogonal_decomp.png")


def main():
    """Generate all geometric PCA figures"""
    print("\n" + "=" * 60)
    print("Generating PCA Geometric Figures")
    print("=" * 60 + "\n")

    generate_3d_projection_figure()
    generate_orthogonal_decomposition_figure()

    print("\n" + "=" * 60)
    print("Successfully generated 2 geometric figures")
    print("=" * 60)
    print("\nFiles created in ../figures/ directory:")
    print("  1. slide25_projection_3d.png       - Slide 25: 3D Projection (300 DPI)")
    print("  2. slide30_orthogonal_decomp.png   - Slide 30: Orthogonal Decomposition (300 DPI)")
    print("\nThese files are ready to use in your LaTeX presentation.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
