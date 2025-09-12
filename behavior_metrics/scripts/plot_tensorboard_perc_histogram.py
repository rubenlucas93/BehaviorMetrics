from matplotlib.ticker import FormatStrFormatter
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# ---------------------------------- Useless


# def plot_histogram_with_percentages_same_transparent(values, algorithm, color, bins=100, format='%.2f', ax=None):
#     # Flatten the list of values
#     values = np.concatenate(values)
#
#     # Plot histogram with percentages
#     first_column = [row[0] for row in values]
#     count, bin_edges = np.histogram(first_column, bins=bins)
#     # _, bin_vis = np.histogram(first_column, bins=x_bins)
#
#     # total_count = sum(count)
#     # percentages = (count / total_count) * 100 # Avoid division by zero
#
#     # Format x-axis to show 2 decimals
#     if ax is None:
#         ax = plt.gca()  # If no ax is provided, use the current axis
#
#     ax.xaxis.set_major_formatter(FormatStrFormatter(format))
#
#     values, _, _ = ax.hist(
#         first_column,
#         bins=bin_edges,
#         histtype='step',  # Shows only the outline
#         edgecolor=color,
#         linewidth=1.5,  # Adjust line thickness
#         weights=np.ones_like(first_column) * 100 / len(first_column),
#         label=algorithm  # Required for legend
#     )
#     return values

# ---------------------------------- Useful

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

    values, _, _ = ax.hist(first_column, bins=bin_edges, edgecolor='black',
        weights = np.ones_like(first_column) * 100 / len(first_column))

    max_y = np.max(values)
    min_x = np.min(first_column)
    max_x = np.max(first_column)

    ax.set_ylabel('Percentage')
    ax.set_title(f'Histogram of {param} with Percentages')

    return max_y, min_x, max_x

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

def extract_csv_data(log_dir, tag):
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


def plot_histogram_comparison_for_metric(axs, metric, x_label, comp_1, comp_2, comp_3, x_bins,
                                         multiplier=1, format='%f', x_axis=None):
    """
    A helper function to plot histograms for the given metric on the provided axes.
    """
    fontsize = 16

    _, values_sac = extract_tensor_data(comp_1['log_dir'], metric)
    max_y_1, min_x_1, max_x_1 = plot_histogram_with_percentages(np.array(values_sac) * multiplier, metric, x_axis=x_axis, x_bins=x_bins, ax=axs[0], format=format)

    axs[0].set_title(comp_1['tag'])
    axs[0].set_xlabel(x_label, fontsize=fontsize)

    # Plot for DDPG
    _, values_ddpg = extract_tensor_data(comp_2['log_dir'], metric)
    max_y_2, min_x_2, max_x_2 = plot_histogram_with_percentages(np.array(values_ddpg) * multiplier, metric, x_axis=x_axis, x_bins=x_bins, ax=axs[1], format=format)

    axs[1].set_title(comp_2['tag'])
    axs[1].set_xlabel(x_label, fontsize=fontsize)

    # Plot for PPO
    _, values_ppo = extract_tensor_data(comp_3['log_dir'], metric)
    max_y_3, min_x_3, max_x_3 = plot_histogram_with_percentages(np.array(values_ppo) * multiplier, metric, x_axis=x_axis, x_bins=x_bins, ax=axs[2], format=format)

    axs[2].set_title(comp_3['tag'])
    axs[2].set_xlabel(x_label, fontsize=fontsize)

    max_y = max(max_y_1, max_y_2,  max_y_3)
    min_x = min(min_x_1, min_x_2, min_x_3)
    max_x = max(max_x_1, max_x_2, max_x_3)

    if x_axis is None:
        x_axis = [min_x * 0.9, max_x * 1.1]  # Add a small buffer
    y_axis = [0, max_y * 1.1]  # Add a small buffer for better visualization

    for ax in axs:
        ax.set_xlim(x_axis)
        ax.set_ylim(y_axis)
        if min_x < 0 < max_x:
            # Ensure 0 is one of the ticks by building symmetric ticks around it
            span = max(x_axis[0], x_axis[1])
            ticks = np.linspace(-span, span, 11)
            ticks = ticks[(ticks >= min_x) & (ticks <= max_x)]
        else:
            # Default to simple linspace if 0 is out of range
            ticks = np.linspace(x_axis[0], x_axis[1], 11)
        ticks = np.where(np.isclose(ticks, 0.0), 0.0, ticks)
        ax.set_xticks(ticks)
        ax.set_yticks(np.linspace(y_axis[0], y_axis[1], 11))


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

    fontsize = 16

    # Adjust the y-axis limits dynamically
    if x_axis is None:
        min_x = min(np.min(sac_min), np.min(ddpg_min), np.min(ppo_min))
        max_x = max(np.max(sac_max), np.max(ddpg_max), np.max(ppo_max))
        x_axis = [min_x * 0.9, max_x * 1.1]  # Add a small buffer
    else:
        min_x = x_axis[0]
        max_x = x_axis[1]

    y_axis = [0, max_y * 1.1]  # Add a small buffer for better visualization
    axs.legend(fontsize=fontsize, bbox_to_anchor=(1, 1))
    axs.set_xlim(x_axis)
    axs.set_ylim(y_axis)

    if min_x < 0 < max_x:
        # Ensure 0 is one of the ticks by building symmetric ticks around it
        span = max(abs(min_x), abs(max_x))
        ticks = np.linspace(-span, span, 11)
        ticks = ticks[(ticks >= min_x) & (ticks <= max_x)]
    else:
        # Default to simple linspace if 0 is out of range
        ticks = np.linspace(min_x, max_x, 11)
    ticks = np.where(np.isclose(ticks, 0.0), 0.0, ticks)

    axs.set_xticks(ticks)
    axs.set_yticks(np.linspace(y_axis[0], y_axis[1], 11))
    axs.tick_params(axis='both', labelsize=fontsize)  # Adjust tick label size
    axs.set_ylabel('percentages', fontsize=fontsize)
    axs.set_xlabel(x_label, fontsize=fontsize)

def plot_histograms_comparison_separated(comp_1, comp_2, comp_3):
    # Create a figure for distances with 3 subplots (one for each log_dir)
    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Comparison of Distances')
    plot_histogram_comparison_for_metric(axs, 'distances', 'distances in pixels (total 512 pixels))', comp_1, comp_2, comp_3, x_axis=[-0.2, 0.2], multiplier=256, x_bins=10, format='%.3f')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust title to fit
    plt.show()
    # plt.savefig('distances_comparison.png')  # Uncomment to save the plot

    # Create a figure for speed with 3 subplots (one for each log_dir)
    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Comparison of Speed')
    plot_histogram_comparison_for_metric(axs, 'speed', 'speed (km/h)', comp_1, comp_2, comp_3, x_axis=[0, 100], x_bins=10, multiplier=3.6, format='%d')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust title to fit
    plt.show()
    # plt.savefig('speed_comparison.png')  # Uncomment to save the plot

def plot_histogram_from_metric_lists(axs, metric, x_label, comp_1, comp_2=None, comp_3=None, x_axis=None, x_bins=10,
                                     multiplier=1):
    """
    Plot histograms for precomputed list-based metrics like 'speeds'.
    comp_2 and comp_3 are optional.
    """
    fontsize = 10
    histograms = []
    min_vals = []
    max_vals = []
    max_y_vals = []

    # Helper to extract and plot one histogram
    def process_comp(comp, color):
        values = np.array(comp['metrics']) * multiplier
        values = values.reshape(-1, 1)  # Nx1 array
        values_hist, min_val, max_val = plot_histogram_with_percentages_same([values], comp['tag'], color, ax=axs)
        histograms.append(values_hist)
        min_vals.append(min_val)
        max_vals.append(max_val)
        max_y_vals.append(np.max(values_hist))

    # Always process comp_1
    process_comp(comp_1, "blue")
    if comp_2 is not None:
        process_comp(comp_2, "red")
    if comp_3 is not None:
        process_comp(comp_3, "black")

    # Axis settings
    if x_axis is None:
        min_x = min(min_vals)
        max_x = max(max_vals)
        x_axis = [min_x * 0.9, max_x * 1.1]
    else:
        min_x, max_x = x_axis

    max_y = max(max_y_vals) if max_y_vals else 1

    axs.set_xlim(x_axis)
    axs.set_ylim([0, max_y * 1.1])
    axs.legend(fontsize=fontsize, bbox_to_anchor=(1, 1))

    ticks = np.linspace(min_x, max_x, 11)
    ticks = np.where(np.isclose(ticks, 0.0), 0.0, ticks)
    axs.set_xticks(ticks)
    axs.set_yticks(np.linspace(0, max_y * 1.1, 11))

    axs.tick_params(axis='both', labelsize=fontsize)
    axs.set_ylabel('percentages', fontsize=fontsize)
    axs.set_xlabel(x_label, fontsize=fontsize)


def plot_histograms_comparison_same(comp_1, comp_2, comp_3):
    # Create a figure for distances with 3 subplots (one for each log_dir)
    fig, axs = plt.subplots(1, 1, figsize=(15, 12))
    # plot_histogram_comparison_in_same_plot(axs, 'distances', 'distances in pixels (total 512 pixels))', comp_1, comp_2, comp_3, x_axis=[-50, 50], multiplier=256, x_bins=10, format='%.3f')
    plot_histogram_comparison_in_same_plot(axs, 'distances', 'distances in cm', comp_1, comp_2, comp_3, x_bins=10, format='%.3f')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust title to fit
    plt.show()
    # plt.savefig('distances_comparison.png')  # Uncomment to save the plot

    # Create a figure for speed with 3 subplots (one for each log_dir)
    fig, axs = plt.subplots(1, 1, figsize=(15, 12))
    plot_histogram_comparison_in_same_plot(axs, 'speed', 'speed (km/h)', comp_1, comp_2, comp_3, x_axis=[32, 80], x_bins=10, multiplier=3.6, format='%d')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust title to fit
    plt.show()

    # fig, axs = plt.subplots(1, 1, figsize=(15, 12))
    # plot_histogram_comparison_in_same_plot(axs, 'curvatures', 'curvature indicator', comp_1, comp_2, comp_3, x_axis=[0, 100], x_bins=10, format='%f')
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.85)  # Adjust title to fit
    # plt.show()
    # plt.savefig('speed_comparison.png')  # Uncomment to save the plot

import time
import os
from datetime import datetime

def save_histograms_comparison_same(comp_1, comp_2, comp_3):
    experiment_starting_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_dir = f'tensorboard_comparisons_{experiment_starting_time}'
    os.makedirs(output_dir, exist_ok=True)

    # Create a figure for distances with 3 subplots (one for each log_dir)
    fig, axs = plt.subplots(1, 1, figsize=(15, 12))
    plot_histogram_comparison_in_same_plot(axs, 'distances', 'distances in pixels (total 512 pixels))', comp_1, comp_2, comp_3, x_axis=[-50, 50], multiplier=256, x_bins=10, format='%.3f')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust title to fit
    plt.savefig(f"tensorboard_comparisons_{experiment_starting_time}/distances_comparison.png")  # Uncomment to save the plot

    # Create a figure for speed with 3 subplots (one for each log_dir)
    fig, axs = plt.subplots(1, 1, figsize=(15, 12))
    plot_histogram_comparison_in_same_plot(axs, 'speed', 'speed (km/h)', comp_1, comp_2, comp_3, x_axis=[32, 80], x_bins=10, multiplier=3.6, format='%d')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust title to fit
    plt.savefig(f"tensorboard_comparisons_{experiment_starting_time}/speed_comparison.png")  # Uncomment to save the plot

    # fig, axs = plt.subplots(1, 1, figsize=(15, 12))
    # plot_histogram_comparison_in_same_plot(axs, 'curvatures', 'curvature indicator', comp_1, comp_2, comp_3, x_axis=[0, 100], x_bins=10, format='%f')
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.85)  # Adjust title to fit
    # plt.show()
    # plt.savefig('curveatures.png')  # Uncomment to save the plot

# SAC entre si
# comp_1 = {
#     'log_dir': [
#         '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/town03_training_town10_inference/sac/20250512-220036',
#         '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/town03_training_town10_inference/sac/20250512-220420'
#     ],
#     'tag': 'SAC_Town03'
# }
# comp_2 = {
#     'log_dir': [
#         '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/town04_training_town10_inference/sac/20250512-210910_town05_anticlock',
#         '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/town04_training_town10_inference/sac/20250512-211430_clock'
#     ],
#     'tag': 'SAC_Town04'
# }
# comp_3 = {
#     'log_dir': [
#         '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/town03Andtown04_training_town10_inference/sac/20250512-214342',
#         '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/town03Andtown04_training_town10_inference/sac/20250512-214703'
#     ],
#     'tag': 'SAC_Both'
# }


# PPO entre si
# comp_1 = {
#     'log_dir': [
#         '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/town03_training_town10_inference/ppo/20250512-232314',
#         '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/town03_training_town10_inference/ppo/20250512-232734'
#     ],
#     'tag': 'PPO_Town03'
# }
# comp_2 = {
#     'log_dir': [
#         '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/town04_training_town10_inference/ppo/20250512-213855',
#         '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/town04_training_town10_inference/ppo/20250512-215325'
#     ],
#     'tag': 'PPO_Town04'
# }
# comp_3 = {
#     'log_dir': [
#         '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/town03Andtown04_training_town10_inference/ppo/20250512-231634',
#         '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/town03Andtown04_training_town10_inference/ppo/20250512-231926'
#     ],
#     'tag': 'PPO_Both'
# }

# Town 04

# comp_1 = {
#     'log_dir': [
#         '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/town04_training_town10_inference/ddpg/20250512-210346_town05_anticlock',
#         '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/town04_training_town10_inference/ddpg/20250512-211204_town10_clock'
#     ],
#     'tag': 'DDPG'
# }
# comp_2 = {
#     'log_dir': [
#         '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/town04_training_town10_inference/ppo/20250512-215325',
#         '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/town04_training_town10_inference/ppo/20250512-213855'
#     ],
#     'tag': 'PPO'
# }
# comp_3 = {
#     'log_dir': [
#         '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/town04_training_town10_inference/sac/20250512-210910_town05_anticlock',
#         '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/town04_training_town10_inference/sac/20250512-211430_clock'
#     ],
#     'tag': 'SAC'
# }


# DDPG entre si

# comp_1 = {
#     'log_dir': [
#         '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/Tensorboard/ddpg/20250515-201059',
#         '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/Tensorboard/ddpg/20250515-201335'
#     ],
#     'tag': 'DDPG'
# }
# comp_2 = {
#     'log_dir': [
#         '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/Tensorboard/sac/20250512-210346',
#         '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/Tensorboard/sac/20250512-211204'
#     ],
#     'tag': 'SAC'
# }
# comp_3 = {
#     'log_dir': [
#         '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/Tensorboard/ppo/20250515-201059',
#         '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/Tensorboard/ppo/20250515-201335'
#     ],
#     'tag': 'PPO'
# }
#
#
# # _______________
#
# plot_histograms_comparison_same(comp_1, comp_2, comp_3)
#plot_histograms_comparison_separated(comp_1, comp_2, comp_3)
