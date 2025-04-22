from matplotlib.ticker import FormatStrFormatter
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def plot_histogram_with_percentages(values, param, x_axis=[-1, 1], y_axis=[0, 100], bins=100, x_bins=15, format='%.2f', ax=None):
    # Flatten the list of values
    values = np.concatenate(values)

    # Plot histogram with percentages
    first_column = [row[0] for row in values]
    count, bin_edges = np.histogram(first_column, bins=bins)
    # _, bin_vis = np.histogram(first_column, bins=x_bins)

    # total_count = sum(count)
    # percentages = (count / total_count) * 100 # Avoid division by zero

    # Format x-axis to show 2 decimals
    if ax is None:
        ax = plt.gca()  # If no ax is provided, use the current axis

    ax.xaxis.set_major_formatter(FormatStrFormatter(format))

    ax.hist(first_column, bins=bin_edges, edgecolor='black',
        weights = np.ones_like(first_column) * 100 / len(first_column))


    ax.set_xlim(x_axis)
    ax.set_ylim(y_axis)
    ax.set_xlabel(param)
    ax.set_ylabel('Percentage')
    ax.set_title(f'Histogram of {param} with Percentages')
    # ax.xticks(bin_vis)
    ax.set_xticks(np.linspace(x_axis[0], x_axis[1], 11))
    ax.set_yticks(np.linspace(y_axis[0], y_axis[1], 11))


def plot_histogram_with_percentages_same_transparent(values, algorithm, color, bins=100, format='%.2f', ax=None):
    # Flatten the list of values
    values = np.concatenate(values)

    # Plot histogram with percentages
    first_column = [row[0] for row in values]
    count, bin_edges = np.histogram(first_column, bins=bins)
    # _, bin_vis = np.histogram(first_column, bins=x_bins)

    # total_count = sum(count)
    # percentages = (count / total_count) * 100 # Avoid division by zero

    # Format x-axis to show 2 decimals
    if ax is None:
        ax = plt.gca()  # If no ax is provided, use the current axis

    ax.xaxis.set_major_formatter(FormatStrFormatter(format))

    values, _, _ = ax.hist(
        first_column,
        bins=bin_edges,
        histtype='step',  # Shows only the outline
        edgecolor=color,
        linewidth=1.5,  # Adjust line thickness
        weights=np.ones_like(first_column) * 100 / len(first_column),
        label=algorithm  # Required for legend
    )
    return values


def plot_histogram_with_percentages_same(values, algorithm, color, bins=100, format='%.2f', ax=None):
    # Flatten the list of values
    values = np.concatenate(values)

    # Plot histogram with percentages
    first_column = [row[0] for row in values]
    count, bin_edges = np.histogram(first_column, bins=bins)
    # _, bin_vis = np.histogram(first_column, bins=x_bins)

    # total_count = sum(count)
    # percentages = (count / total_count) * 100 # Avoid division by zero

    # Format x-axis to show 2 decimals
    if ax is None:
        ax = plt.gca()  # If no ax is provided, use the current axis

    ax.xaxis.set_major_formatter(FormatStrFormatter(format))

    values, _, _ = ax.hist(
        first_column,
        bins=bin_edges,
        histtype='bar',  # Shows only the outline
        color=color,
        alpha=0.3,
        linewidth=1.5,  # Adjust line thickness
        weights=np.ones_like(first_column) * 100 / len(first_column),
        label=algorithm  # Required for legend
    )


    return values, np.min(first_column), np.max(first_column)

def extract_tensor_data(log_dir, tag):
    # Handle both single and multiple log directories
    if isinstance(log_dir, str):
        log_dirs = [log_dir]
    else:
        log_dirs = log_dir

    all_steps = []
    all_values = []

    for dir_path in log_dirs:
        ea = event_accumulator.EventAccumulator(dir_path, size_guidance={event_accumulator.TENSORS: 0})
        ea.Reload()

        print(f"Available tags in '{dir_path}':", ea.Tags())

        if tag not in ea.Tags().get('tensors', []):
            raise KeyError(f"Key '{tag}' was not found in '{dir_path}'")

        tensors = ea.Tensors(tag)

        for event in tensors:
            all_steps.append(event.step)
            tensor_values = tf.make_ndarray(event.tensor_proto)
            all_values.append(tensor_values)

    return all_steps, all_values


def print_available_tags(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    print("Available tags in the log directory:")
    for tag_type, tags in ea.Tags().items():
        print(f"{tag_type}:")
        for tag in tags:
            print(f" - {tag}")


def plot_histogram(log_dir, param, x_bins=15):
    try:
        steps, values = extract_tensor_data(log_dir, param)
        # print(f"Steps: {steps}")
        # print(f"Values: {values}")
        plot_histogram_with_percentages(values, param, x_bins=x_bins)
    except KeyError as e:
        print(e)


# Function to plot both histograms side by side
def plot_histograms_side_by_side(log_dir, param1, param2):
    plt.figure(figsize=(12, 6))  # Adjust figure size as needed

    # Plot first histogram
    plt.subplot(1, 2, 1)
    plot_histogram(log_dir, param1)

    # Plot second histogram
    plt.subplot(1, 2, 2)
    plot_histogram(log_dir, param2)

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()

# Function to plot both histograms side by side
def plot_one_histogram(param, x_bins):
    plot_histogram(param, x_bins)
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()


def plot_all_histograms(log_dir):
    plot_histograms_side_by_side(log_dir, 'actions_v', 'actions_w')

    _, values = extract_tensor_data(log_dir, 'distances')
    plot_histogram_with_percentages(np.array(values), 'distances', x_axis=[-0.2, 0.2], x_bins=10)
    plt.show()
    # plt.savefig('distances_histogram.png')  # Save the plot as a PNG file

    _, values = extract_tensor_data(log_dir, 'speed')
    plot_histogram_with_percentages(np.array(values) * 3.6, 'speed', x_axis=[1, 100], x_bins=10, format='%d')
    plt.show()
    # plt.savefig('speed_histogram.png')  # Save the plot as a PNG file


def plot_histogram_comparison_for_metric(axs, metric, log_dir_sac, log_dir_ddpg, log_dir_ppo, x_axis, x_bins, multiplier=1, format='%f'):
    """
    A helper function to plot histograms for the given metric on the provided axes.
    """
    # Plot for SAC
    _, values_sac = extract_tensor_data(log_dir_sac, metric)
    plot_histogram_with_percentages(np.array(values_sac) * multiplier, metric, x_axis=x_axis, x_bins=x_bins, ax=axs[0], format=format)
    axs[0].set_title('SAC')

    # Plot for DDPG
    _, values_ddpg = extract_tensor_data(log_dir_ddpg, metric)
    plot_histogram_with_percentages(np.array(values_ddpg) * multiplier, metric, x_axis=x_axis, x_bins=x_bins, ax=axs[1], format=format)
    axs[1].set_title('DDPG')

    # Plot for PPO
    _, values_ppo = extract_tensor_data(log_dir_ppo, metric)
    plot_histogram_with_percentages(np.array(values_ppo) * multiplier, metric, x_axis=x_axis, x_bins=x_bins, ax=axs[2], format=format)
    axs[2].set_title('PPO')

def plot_histogram_comparison_in_same_plot(axs, metric, x_label, comp_1, comp_2, comp_3, x_axis, x_bins,
                                         multiplier=1, format='%f'):
    """
    A helper function to plot histograms for the given metric on the provided axes.
    """
    _, values_sac = extract_tensor_data(comp_1['log_dir'], metric)
    values_hist_sac, sac_min, sac_max = plot_histogram_with_percentages_same(np.array(values_sac) * multiplier, comp_1['tag'], "blue", ax=axs, format=format)
    _, values_ddpg = extract_tensor_data(comp_2['log_dir'], metric)
    values_hist_ddpg, ddpg_min, ddpg_max = plot_histogram_with_percentages_same(np.array(values_ddpg) * multiplier, comp_2['tag'], "red", ax=axs, format=format)
    _, values_ppo = extract_tensor_data(comp_3['log_dir'], metric)
    values_hist_ppo, ppo_min, ppo_max = plot_histogram_with_percentages_same(np.array(values_ppo) * multiplier, comp_3['tag'], "black", ax=axs, format=format)

    max_y = max(np.max(values_hist_sac), np.max(values_hist_ddpg),  np.max(values_hist_ppo))
    min_x = min(np.min(sac_min), np.min(ddpg_min), np.min(ppo_min))
    max_x = max(np.max(sac_max), np.max(ddpg_max), np.max(ppo_max))

    fontsize = 16

    # Adjust the y-axis limits dynamically
    x_axis = [min_x * 0.9, max_x * 1.1]  # Add a small buffer
    y_axis = [0, max_y * 1.1]  # Add a small buffer for better visualization
    axs.legend(fontsize=fontsize, bbox_to_anchor=(1, 1))
    axs.set_xlim(x_axis)
    axs.set_ylim(y_axis)
    # ax.xticks(bin_vis)
    axs.set_xticks(np.linspace(x_axis[0], x_axis[1], 11))
    axs.set_yticks(np.linspace(y_axis[0], y_axis[1], 11))
    axs.tick_params(axis='both', labelsize=fontsize)  # Adjust tick label size
    axs.set_ylabel('percentages', fontsize=fontsize)
    axs.set_xlabel(x_label, fontsize=fontsize)

def plot_histograms_comparison(log_dir_sac, log_dir_ddpg, log_dir_ppo):
    # Create a figure for distances with 3 subplots (one for each log_dir)
    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Comparison of Distances')
    plot_histogram_comparison_for_metric(axs, 'distances', log_dir_sac, log_dir_ddpg, log_dir_ppo, x_axis=[-0.2, 0.2], x_bins=10)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust title to fit
    plt.show()
    # plt.savefig('distances_comparison.png')  # Uncomment to save the plot

    # Create a figure for speed with 3 subplots (one for each log_dir)
    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Comparison of Speed')
    plot_histogram_comparison_for_metric(axs, 'speed (km/h)', log_dir_sac, log_dir_ddpg, log_dir_ppo, x_axis=[0, 100], x_bins=10, multiplier=3.6, format='%d')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust title to fit
    plt.show()
    # plt.savefig('speed_comparison.png')  # Uncomment to save the plot

def plot_histograms_comparison_same(comp_1, comp_2, comp_3):
    # Create a figure for distances with 3 subplots (one for each log_dir)
    fig, axs = plt.subplots(1, 1, figsize=(15, 12))
    plot_histogram_comparison_in_same_plot(axs, 'distances', 'distances (pixels normalized [-1, 1])', comp_1, comp_2, comp_3, x_axis=[-0.2, 0.2], x_bins=10, format='%.3f')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust title to fit
    plt.show()
    # plt.savefig('distances_comparison.png')  # Uncomment to save the plot

    # Create a figure for speed with 3 subplots (one for each log_dir)
    fig, axs = plt.subplots(1, 1, figsize=(15, 12))
    plot_histogram_comparison_in_same_plot(axs, 'speed', 'speed (km/h)', comp_1, comp_2, comp_3, x_axis=[0, 100], x_bins=10, multiplier=3.6, format='%d')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust title to fit
    plt.show()

    fig, axs = plt.subplots(1, 1, figsize=(15, 12))
    plot_histogram_comparison_in_same_plot(axs, 'curvatures', 'curvature indicator', comp_1, comp_2, comp_3, x_axis=[0, 100], x_bins=10, format='%f')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust title to fit
    plt.show()
    # plt.savefig('speed_comparison.png')  # Uncomment to save the plot

# Entrenados en town04
# comp_1 = {
#     'log_dir': '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/Tensorboard/sac/20250404-200346',
#     'tag': 'SAC'
# }
# comp_2 = {
#     'log_dir': '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/Tensorboard/ddpg/20250407-201838',
#     'tag': 'DDPG'
# }
# comp_3 = {
#     'log_dir': '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/Tensorboard/ppo/20250404-202616',
#     'tag': 'PPO'
# }
# ____________________

# Entrenados en town03
# comp_1 = {
#     'log_dir': '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/town03_training_town04_inference/sac_20250404-205644',
#     'tag': 'SAC'
# }
# comp_2 = {
#     'log_dir': '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/town03_training_town04_inference/ddpg_20250404-210104',
#     'tag': 'DDPG'
# }
# comp_3 = {
#     'log_dir': '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/town03_training_town04_inference/ppo/20250409-203604',
#     'tag': 'PPO'
# }

# Current

# comp_1 = {
#     'log_dir': '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/town03Andtown04_training_town10_inference/ppo/20250410-225944',
#     'tag': 'PPO_Both'
# }
# comp_2 = {
#     'log_dir': '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/town04_training_town10_inference/ppo_20250404-202616',
#     'tag': 'PPO_Town04'
# }
# comp_3 = {
#     'log_dir': '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/best_logs/town03_training_town10_inference/ppo/20250409-203945',
#     'tag': 'PPO_Town03'
# }

comp_1 = {
    'log_dir': [
        '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/Tensorboard/sac/20250419-103209',
        # '',
    ],
    'tag': 'PPO_Both'
}
comp_2 = {
    'log_dir': [
        '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/Tensorboard/sac/20250419-103209',
        # '',
    ],
    'tag': 'PPO_Town04'
}
comp_3 = {
    'log_dir': [
        '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/Tensorboard/sac/20250419-103209',
        # '',
    ],
    'tag': 'PPO_Town03'
}
# _______________

plot_histograms_comparison_same(comp_1, comp_2, comp_3)
