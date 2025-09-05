"""
Utility functions for the RAG walkthrough notebook.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_triangular_similarity_heatmap(similarity_matrix, labels, title="Semantic Similarity Between Different Queries", 
                                     figsize=(10, 8), colormap='coolwarm_r'):
    """
    Create a compact triangular heatmap visualization for a similarity matrix.
    
    Args:
        similarity_matrix (np.ndarray): Square similarity matrix
        labels (list): List of labels for the queries/items
        title (str): Title for the plot
        figsize (tuple): Figure size (width, height)
        colormap (str): Matplotlib colormap name
    
    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    # Create a more compact triangular visualization
    fig, ax = plt.subplots(figsize=figsize)

    # Create custom triangular heatmap
    n = len(labels)
    for i in range(n):
        for j in range(i+1):
            value = similarity_matrix[i, j]
            # Use colormap scaling
            normalized_value = (value - similarity_matrix.min()) / (similarity_matrix.max() - similarity_matrix.min())
            color = plt.cm.get_cmap(colormap)(normalized_value)
            
            # Create rectangular patch for each cell
            rect = plt.Rectangle((j, n-1-i), 1, 1, facecolor=color, edgecolor='white', linewidth=1)
            ax.add_patch(rect)
            
            # Add text annotation
            text_color = 'black' if normalized_value > 0.5 else 'white'
            ax.text(j + 0.5, n-1-i + 0.5, f'{value:.3f}', 
                    ha='center', va='center', fontsize=10, color=text_color)

    # Set up the plot
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect('equal')

    # Set labels
    x_labels = [labels[i] for i in range(n)]
    y_labels = [labels[n-1-i] for i in range(n)]

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_yticklabels(y_labels, rotation=0)

    # Remove extra triangular area
    for i in range(n):
        for j in range(i+1, n):
            # Hide upper triangle by setting it to background color
            rect = plt.Rectangle((j, n-1-i), 1, 1, facecolor='white', edgecolor='white')
            ax.add_patch(rect)

    ax.set_title(title, fontsize=14, pad=20)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, 
                              norm=plt.Normalize(vmin=similarity_matrix.min(), vmax=similarity_matrix.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=20)

    plt.tight_layout()
    
    return fig, ax
