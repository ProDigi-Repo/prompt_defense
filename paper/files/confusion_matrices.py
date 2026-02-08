import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

# Confusion matrices data (values in percentages)
# Format: [[top-left, top-right], [bottom-left, bottom-right]]
matrices = {
    'L_50': np.array([[77, 1], [23, 99]]),
    'CS_50': np.array([[96, 78], [4, 22]]),
    'L_90': np.array([[2, 0], [98, 100]]),
    'CS_90': np.array([[19, 2], [81, 98]])
}

# Create figure with 4 subplots - make more rectangular for A4 width
fig = plt.figure(figsize=(14, 2.5))

# Define subplot positions with extra space between groups
# Group 1 (L): positions 1-2, Group 2 (CS): positions 3-4
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1], 
                      wspace=0.15, hspace=0.1)

# Within-group spacing vs between-group spacing
axes = [
    fig.add_subplot(gs[0, 0]),  # L, threshold 50
    fig.add_subplot(gs[0, 1]),  # L, threshold 90
    fig.add_subplot(gs[0, 2]),  # CS, threshold 50
    fig.add_subplot(gs[0, 3])   # CS, threshold 90
]

matrix_keys = ['L_50', 'L_90', 'CS_50', 'CS_90']
titles = ['L / 50%', 'L / 90%', 'CS / 50%', 'CS / 90%']

for idx, (ax, key, title) in enumerate(zip(axes, matrix_keys, titles)):
    matrix = matrices[key]
    
    # Display matrix with grayscale (0% = white, 100% = black)
    im = ax.imshow(matrix, cmap='gray', vmin=0, vmax=100, aspect='auto')
    
    # Add text annotations with percentage values
    for i in range(2):
        for j in range(2):
            value = matrix[i, j]
            # Use white text for cells > 50%, black for cells <= 50%
            text_color = 'white' if value < 50 else 'black'
            text = ax.text(j, i, f'{value}%',
                          ha='center', va='center',
                          color=text_color, fontsize=14, fontweight='bold')
    
    # Set title below the matrix
    ax.set_xlabel(title, fontsize=12, fontweight='bold', labelpad=10)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add subtle borders
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(1)

# Manually adjust spacing to create clear groups
# Move CS group further right
pos1 = axes[0].get_position()
pos2 = axes[1].get_position()
pos3 = axes[2].get_position()
pos4 = axes[3].get_position()

# Increase gap between groups (between axes[1] and axes[2])
gap_within = pos2.x0 - pos1.x1  # gap within group
gap_between = gap_within * 2.5  # larger gap between groups

# Reposition CS group
shift = gap_between - (pos3.x0 - pos2.x1)
axes[2].set_position([pos3.x0 + shift, pos3.y0, pos3.width, pos3.height])
axes[3].set_position([pos4.x0 + shift, pos4.y0, pos4.width, pos4.height])

plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("Confusion matrices chart saved to confusion_matrices.png")

plt.show()
