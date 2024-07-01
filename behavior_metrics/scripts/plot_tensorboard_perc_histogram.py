from matplotlib.ticker import FormatStrFormatter
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def plot_histogram_with_percentages(values, param, bins=100):
    # Flatten the list of values
    values = np.concatenate(values)

    # Plot histogram with percentages
    first_column = [row[0] for row in values]
    counts, bin_edges = np.histogram(first_column, bins=bins)
    _, bin_vis = np.histogram(first_column, bins=15)

    # Format x-axis to show 2 decimals
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.hist(first_column, bins=bin_edges, edgecolor='black')
    plt.xlabel(param)
    plt.ylabel('Percentage')
    plt.title(f'Histogram of {param} with Percentages')
    plt.xticks(bin_vis)

def extract_tensor_data(log_dir, tag):
    # Load the event accumulator
    ea = event_accumulator.EventAccumulator(log_dir, size_guidance={event_accumulator.TENSORS: 0})
    ea.Reload()

    # Print available tags for verification
    print("Available tags:", ea.Tags())

    if tag not in ea.Tags()['tensors']:
        raise KeyError(f"Key '{tag}' was not found in Reservoir")

    # Extract tensor events
    tensors = ea.Tensors(tag)

    steps = []
    values = []

    for event in tensors:
        steps.append(event.step)
        tensor_values = tf.make_ndarray(event.tensor_proto)  # Extract the tensor values

        # # Print tensor values for debugging
        # print(f"Step: {event.step}, Values: {tensor_values}")

        values.append(tensor_values)

    return steps, values


def print_available_tags(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    print("Available tags in the log directory:")
    for tag_type, tags in ea.Tags().items():
        print(f"{tag_type}:")
        for tag in tags:
            print(f" - {tag}")


def plot_histogram(param):
    try:
        steps, values = extract_tensor_data(log_dir, param)
        # print(f"Steps: {steps}")
        # print(f"Values: {values}")
    except KeyError as e:
        print(e)

    plot_histogram_with_percentages(values, param)

# Function to plot both histograms side by side
def plot_histograms_side_by_side(param1, param2):
    plt.figure(figsize=(12, 6))  # Adjust figure size as needed

    # Plot first histogram
    plt.subplot(1, 2, 1)
    plot_histogram(param1)

    # Plot second histogram
    plt.subplot(1, 2, 2)
    plot_histogram(param2)

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()


log_dir = '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/logs/Tensorboard/ppo/20240704-165223'
plot_histograms_side_by_side('actions_v', 'actions_w')


