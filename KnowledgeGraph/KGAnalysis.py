import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

def create_publication_visualizations():
    # Set style for publication
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    
    # Get plasma color palette
    plasma_colors = sns.color_palette("plasma", n_colors=6)
    
    # Node distribution data
    node_data = {
        'SMILES': 1686532,
        'Patent': 48823,
        'DTXSID': 10972,
        'CPC': 8619,
        'Functional_Use': 95,
        'Predicted_Use': 38
    }
    
    # Edge distribution data
    edge_data = {
        'predicted_use': 8609969,
        'appears_in_patent': 2879821,
        'has_cpc': 2879821,
        'verified_use': 137326,
        'has_dtxsid': 137326
    }
    
    # Create Figure 1: Summary Statistics
    fig1 = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig1)
    
    # Plot 1: Node Distribution
    ax1 = fig1.add_subplot(gs[0, 0])
    nodes = pd.Series(node_data)
    bars1 = nodes.plot(kind='bar', ax=ax1, logy=True, color=plasma_colors)
    ax1.set_title('Node Type Distribution (Log Scale)')
    ax1.set_ylabel('Count (log)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Edge Distribution
    ax2 = fig1.add_subplot(gs[0, 1])
    edges = pd.Series(edge_data)
    bars2 = edges.plot(kind='bar', ax=ax2, logy=True, color=plasma_colors[1:])
    ax2.set_title('Edge Type Distribution (Log Scale)')
    ax2.set_ylabel('Count (log)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Degree Distribution
    ax3 = fig1.add_subplot(gs[1, :])
    degree_stats = {
        'Average Degree': 16.69,
        'Median Degree': 1.00,
        'Max Degree': 2048121
    }
    
    # Create theoretical power law distribution for comparison
    x = np.logspace(0, 6, 100)
    y = x**(-2)  # Power law with exponent -2
    
    # Plot power law in plasma color
    ax3.loglog(x, y, '-', color=plasma_colors[0], alpha=0.7, 
               label='Theoretical Power Law (α ≈ -2)')
    ax3.set_xlabel('Node Degree (log)')
    ax3.set_ylabel('Frequency (log)')
    ax3.set_title('Degree Distribution (Power Law)')
    ax3.grid(True, which="both", ls="-", alpha=0.2)
    ax3.legend()
    
    # Add degree statistics as text with nice background
    stats_text = '\n'.join([f'{k}: {v:,.2f}' for k, v in degree_stats.items()])
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, 
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', 
                      alpha=0.8, edgecolor=plasma_colors[0]))
    
    plt.tight_layout()
    plt.savefig('kg_summary_stats.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # Create Figure 2: Network Structure Overview
    fig2, ax = plt.subplots(figsize=(12, 8))
    
    # Create a simplified network visualization
    pos = {
        'SMILES': (0.5, 0.5),
        'Uses': (0.8, 0.8),
        'Patents': (0.2, 0.8),
        'CPC': (0.2, 0.2),
        'DTXSID': (0.8, 0.2)
    }
    
    # Draw nodes with plasma colors
    for i, (node, (x, y)) in enumerate(pos.items()):
        size = 5000 if node == 'SMILES' else 2000
        ax.scatter(x, y, s=size, alpha=0.6, label=node, 
                  color=plasma_colors[i])
        ax.text(x, y, f'{node}\n', ha='center', va='center', 
                color='black')
    
    # Draw edges with gradient colors
    for i, (start, end) in enumerate([
        ('SMILES', 'Uses'),
        ('SMILES', 'Patents'),
        ('Patents', 'CPC'),
        ('SMILES', 'DTXSID')
    ]):
        ax.plot([pos[start][0], pos[end][0]], 
                [pos[start][1], pos[end][1]], 
                '-', color=plasma_colors[i], alpha=0.4, linewidth=2)
    
    ax.set_title('Knowledge Graph Structure Overview')
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.savefig('kg_structure_overview.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')

if __name__ == "__main__":
    create_publication_visualizations()