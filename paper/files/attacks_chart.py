import matplotlib.pyplot as plt
import numpy as np

# Data from the experiment (N=17 attacks)
# Format: [temp_0.0, temp_0.7] for models, [CM01, CM02] for guards
# data = {
#     'M01': [5, 5],
#     'M02': [6, 5],
#     'M03': [1, 1],
#     'M04': [5, 6],
#     'M05': [0, 0],
#     'M06': [1, 0],
#     'M07': [2, 2],
#     'M08': [6, 4],
#     'Guards': [8, 6]
# }

data_succ = {
    "M01": [22, 23],
    "M02": [11, 9],
    "M03": [5, 2],
    "M04": [9, 12],
    "M05": [0, 0],
    "M06": [4, 0],
    "M07": [1, 3],
    "M08": [5, 4],
    "Guards": [8, 6],
}

data_unsucc = {
    "M01": [29, 28],
    "M02": [40, 42],
    "M03": [46, 49],
    "M04": [42, 39],
    "M05": [52, 51],
    "M06": [47, 51],
    "M07": [50, 48],
    "M08": [56, 47],
    "Guards": [9, 11],
}

# Prepare data for plotting
model_names = list(data_succ.keys())
successful_attacks = np.array([data_succ[model] for model in model_names])
unsuccessful_attacks = np.array([data_unsucc[model] for model in model_names])

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
        group_start = i * (
            n_bars_per_group * bar_width + within_group_spacing + group_spacing
        )
        positions.extend([group_start, group_start + bar_width + within_group_spacing])

    positions = np.array(positions)

    # Get data for this subset
    successful_subset = successful_attacks[start_idx:end_idx].flatten()
    unsuccessful_subset = unsuccessful_attacks[start_idx:end_idx].flatten()

    # Normalize each bar to 1 (proportion of successful vs unsuccessful)
    successful_normalized = []
    unsuccessful_normalized = []
    for succ, unsucc in zip(successful_subset, unsuccessful_subset):
        total = succ + unsucc
        if total > 0:
            successful_normalized.append(succ / total)
            unsuccessful_normalized.append(unsucc / total)
        else:
            successful_normalized.append(0.0)
            unsuccessful_normalized.append(0.0)

    successful_normalized = np.array(successful_normalized)
    unsuccessful_normalized = np.array(unsuccessful_normalized)

    # Create stacked bars (data_succ below, data_unsucc above)
    bars_bottom = ax.bar(
        positions,
        successful_normalized,
        bar_width,
        color="black",
        label="Successful attacks",
    )
    bars_top = ax.bar(
        positions,
        unsuccessful_normalized,
        bar_width,
        bottom=successful_normalized,
        color="gray",
        label="Unsuccessful attacks",
    )

    # Add white text labels on each segment (showing proportions)
    for i, (pos, succ, unsucc) in enumerate(
        zip(positions, successful_normalized, unsuccessful_normalized)
    ):
        # Label for successful attacks (bottom, black section)
        if succ > 0.05:
            ax.text(
                pos,
                succ / 2,
                f"{succ:.2f}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=10,
            )

        # Label for unsuccessful attacks (top, gray section)
        if unsucc > 0.05:
            ax.text(
                pos,
                succ + unsucc / 2,
                f"{unsucc:.2f}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=10,
            )

    # Set x-axis labels
    group_centers = []
    for i in range(n_groups_subset):
        group_start = i * (
            n_bars_per_group * bar_width + within_group_spacing + group_spacing
        )
        group_center = group_start + (bar_width + within_group_spacing) / 2
        group_centers.append(group_center)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_names, fontsize=11, fontweight="bold")
    ax.xaxis.set_tick_params(pad=20)

    # Add sub-labels for temperature/guard models
    for i, group_name in enumerate(group_names):
        group_start = i * (
            n_bars_per_group * bar_width + within_group_spacing + group_spacing
        )
        bar1_pos = group_start
        bar2_pos = group_start + bar_width + within_group_spacing

        if group_name == "Guards":
            label1, label2 = "CM01", "CM02"
        else:
            label1, label2 = "0.0", "0.7"

        ax.text(bar1_pos, -0.1, label1, ha="center", va="top", fontsize=9)
        ax.text(bar2_pos, -0.1, label2, ha="center", va="top", fontsize=9)

    # Customize the plot
    ax.set_ylabel("Proportion", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# Create plots for both rows
create_bar_plot(ax1, model_names[:first_row_groups], 0, first_row_groups)
create_bar_plot(ax2, model_names[first_row_groups:], first_row_groups, len(model_names))

# Add legend to the top plot only
ax1.legend(loc="upper right")

# Adjust layout to prevent label cutoff for two rows
plt.tight_layout()
plt.subplots_adjust(bottom=0.15, hspace=0.4)

# Save the figure
plt.savefig("attacks_chart.png", dpi=300, bbox_inches="tight")
print("Chart saved to prompt_defense_chart.png")

# Display the plot
plt.show()
