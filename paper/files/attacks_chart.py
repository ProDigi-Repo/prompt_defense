import matplotlib.pyplot as plt
import numpy as np

# Data from the experiment (N=17 attacks)
# Format: [temp_0.0, temp_0.7] for models, [CM01, CM02] for guards
data = {
    'M01': [5, 5],
    'M02': [6, 5],
    'M03': [1, 1],
    'M04': [5, 6],
    'M05': [0, 0],
    'M06': [1, 0],
    'M07': [2, 2],
    'M08': [6, 4],
    'Guards': [8, 6]
}

total_attacks = 17

# Prepare data for plotting
model_names = list(data.keys())
successful_attacks = np.array([data[model] for model in model_names])
unsuccessful_attacks = total_attacks - successful_attacks

# Create figure with 2 rows for better layout
# First row: 4 groups (M01-M04), Second row: remaining groups (M05-M07, M08, Guards)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5))

# Split data into two groups
first_row_groups = 4  # M01-M04
second_row_groups = len(model_names) - first_row_groups  # M05-M07, M08, Guards

def create_bar_plot(ax, group_names, start_idx, end_idx):
    """Create bar plot for a subset of groups"""
    n_groups_subset = len(group_names)
    n_bars_per_group = 2

    # Use absolute fixed width for consistency across both rows
    bar_width = 0.4
    group_spacing = 1.0
    within_group_spacing = 0.2

    # Calculate positions for each bar
    positions = []
    for i in range(n_groups_subset):
        group_start = i * (n_bars_per_group * bar_width + within_group_spacing + group_spacing)
        positions.extend([group_start, group_start + bar_width + within_group_spacing])

    positions = np.array(positions)

    # Get data for this subset
    successful_subset = successful_attacks[start_idx:end_idx].flatten()
    unsuccessful_subset = unsuccessful_attacks[start_idx:end_idx].flatten()

    # Create stacked bars
    bars_bottom = ax.bar(positions, successful_subset, bar_width,
                         color='black', label='Successful attacks')
    bars_top = ax.bar(positions, unsuccessful_subset, bar_width,
                      bottom=successful_subset, color='gray', label='Unsuccessful attacks')

    # Add white text labels on each segment
    for i, (pos, succ, unsucc) in enumerate(zip(positions, successful_subset, unsuccessful_subset)):
        # Label for successful attacks (bottom, black section)
        if succ > 0:
            ax.text(pos, succ/2, str(int(succ)),
                   ha='center', va='center', color='white', fontweight='bold', fontsize=10)

        # Label for unsuccessful attacks (top, gray section)
        if unsucc > 0:
            ax.text(pos, succ + unsucc/2, str(int(unsucc)),
                   ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    # Set x-axis labels
    group_centers = []
    for i in range(n_groups_subset):
        group_start = i * (n_bars_per_group * bar_width + within_group_spacing + group_spacing)
        group_center = group_start + (bar_width + within_group_spacing) / 2
        group_centers.append(group_center)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_names, fontsize=11, fontweight='bold')
    ax.xaxis.set_tick_params(pad=20)

    # Add sub-labels for temperature/guard models
    for i, group_name in enumerate(group_names):
        group_start = i * (n_bars_per_group * bar_width + within_group_spacing + group_spacing)
        bar1_pos = group_start
        bar2_pos = group_start + bar_width + within_group_spacing

        if group_name == 'Guards':
            label1, label2 = 'CM01', 'CM02'
        else:
            label1, label2 = '0.0', '0.7'

        ax.text(bar1_pos, -2.8, label1, ha='center', va='top', fontsize=9)
        ax.text(bar2_pos, -2.8, label2, ha='center', va='top', fontsize=9)

    # Customize the plot
    ax.set_ylabel('#attacks', fontsize=12, fontweight='bold')
    ax.set_ylim(-4, 18)
    ax.set_yticks([0, 8, 16])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('data', 0))

# Create plots for both rows
create_bar_plot(ax1, model_names[:first_row_groups], 0, first_row_groups)
create_bar_plot(ax2, model_names[first_row_groups:], first_row_groups, len(model_names))

# Add legend to the top plot only
ax1.legend(loc='upper right')

# Adjust layout to prevent label cutoff for two rows
plt.tight_layout()
plt.subplots_adjust(bottom=0.25, hspace=0.4)  # Make room for sublabels and space between rows

# Save the figure
plt.savefig('attacks_chart.png', dpi=300, bbox_inches='tight')
print("Chart saved to prompt_defense_chart.png")

# Display the plot
plt.show()
