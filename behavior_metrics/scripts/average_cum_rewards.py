import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf  # Required for converting tensor_proto to numpy array
from scipy.integrate import trapz
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import List, Tuple, Dict, Union, Optional

# Set a style for the plot
plt.style.use('ggplot')


class AUCCalculator:
    """
    A class to recursively find TensorBoard logs, extract scalar/tensor data,
    apply smoothing, average learning curves, calculate AUC, and plot the results.
    """

    # Define the metric tags for each algorithm.
    TAG_MAP = {
        'sac': 'eval/mean_reward',  # Often used in eval callbacks
        'td3': 'rollout/ep_rew_mean',
        'ppo': 'rollout/ep_rew_mean',
        'ddpg': 'rollout/ep_rew_mean',
    }

    # ⚠️ IMPORTANT: Set the custom metric tag here if it's the same for all,
    # overriding the TAG_MAP if needed for this specific run.
    DEFAULT_TAG = 'cum_rewards'

    def __init__(self, base_log_dir: str, smoothing_window: int = 1):
        self.base_log_dir = base_log_dir
        # Store the smoothing window size
        self.smoothing_window = smoothing_window
        # self.all_data stores (steps, values) for every run found
        self.all_data: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {alg: [] for alg in self.TAG_MAP.keys()}
        # self.averaged_curves stores the single mean curve (steps, values) for each alg
        self.averaged_curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _apply_moving_average(values: np.ndarray, window: int) -> np.ndarray:
        """
        Applies a moving average (rolling mean) to a 1D NumPy array.

        Args:
            values: The 1D array of metric values.
            window: The window size (number of steps) for smoothing.

        Returns:
            The smoothed 1D array (truncated by window size).
        """
        if window <= 1 or len(values) < window:
            return values

        # Create a kernel of ones for convolution
        weights = np.ones(window) / window

        # Apply convolution to calculate the moving average
        # mode='valid' ensures we only return values where the window is fully inside the data.
        smoothed = np.convolve(values, weights, mode='valid')

        # Prepend the start of the values (unsmoothed) to keep the length consistent for plotting
        # We prepend the first (window - 1) unsmoothed values
        return np.concatenate([values[:window - 1], smoothed])

    def _find_best_log_dir(self, start_dir: str, required_tag: str) -> Optional[str]:
        """
        Recursively searches for a TensorBoard log directory that contains
        an event file AND contains the required scalar or tensor tag.

        This logic is crucial for robustness against nested log directories.
        """
        checked_dirs = set()

        # If the user passed the full file path, we just use its directory
        if os.path.isfile(start_dir) and 'tfevents' in os.path.basename(start_dir):
            start_dir = os.path.dirname(start_dir)

        for root, _, files in os.walk(start_dir):
            log_dir = root

            # Check if we have already processed this directory
            if log_dir in checked_dirs:
                continue

            # Check if any file in the current directory looks like a TensorBoard event file
            if any('tfevents' in file for file in files):
                checked_dirs.add(log_dir)

                try:
                    # Initialize EventAccumulator with the directory containing the tfevents file
                    # Ensure both scalars and tensors are loaded (size_guidance={'scalars': 0, 'tensors': 0})
                    ea = EventAccumulator(log_dir, size_guidance={'scalars': 0, 'tensors': 0})
                    ea.Reload()

                    # Check if the required tag is present in SCALARS or TENSORS
                    if required_tag in ea.Tags().get('scalars', []) or required_tag in ea.Tags().get('tensors', []):
                        return log_dir  # Found the correct log directory
                except Exception as e:
                    # Ignore logs that fail to load
                    # print(f"Skipping log directory {log_dir} due to error: {e}")
                    continue

        return None

    def _extract_data(self, log_dir: str, tag: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts step numbers and values for a given tag, checking for Scalars
        and then for Tensors. Includes the recursive log finding logic.
        """
        # 1. Find the actual log directory that contains the required data
        actual_log_dir = self._find_best_log_dir(log_dir, tag)

        if not actual_log_dir:
            # If the required tag is not found anywhere in the tree, raise an error
            raise KeyError(f"Key '{tag}' was not found in any valid log file within the path {log_dir}.")

        # 2. Initialize and reload the EventAccumulator using the found directory
        ea = EventAccumulator(actual_log_dir, size_guidance={'scalars': 0, 'tensors': 0})  # Ensure tensors are loaded
        ea.Reload()

        all_steps = []
        all_values = []

        # --- TENSOR LOGIC (Modified to retrieve tensor data) ---
        if tag in ea.Tags().get('tensors', []):
            tensors = ea.Tensors(tag)

            for event in tensors:
                all_steps.append(event.step)
                # Use tf.make_ndarray to convert the tensor_proto to a numpy array
                tensor_values = tf.make_ndarray(event.tensor_proto)

                # Assuming 'cum_rewards' is a single value tensor, extract the single item
                all_values.append(tensor_values.item())

                # Separate steps and values into NumPy arrays
            steps = np.array(all_steps)
            values = np.array(all_values)
            return steps, values

        # --- SCALAR LOGIC (Original) ---
        elif tag in ea.Tags().get('scalars', []):
            scalars = ea.Scalars(tag)

            # Separate steps and values into NumPy arrays
            steps = np.array([s.step for s in scalars])
            values = np.array([s.value for s in scalars])

            return steps, values

        # --- ERROR HANDLING ---
        else:
            available_scalar_tags = ea.Tags().get('scalars', [])
            available_tensor_tags = ea.Tags().get('tensors', [])
            raise KeyError(
                f"Internal Error: Key '{tag}' not found. Available SCALAR tags: {available_scalar_tags}. Available TENSOR tags: {available_tensor_tags}"
            )

    def find_logs_and_extract(self):
        """
        Recursively walks the base directory, finds event files, extracts data,
        and applies moving average smoothing to each run.
        """
        print(f"Starting recursive search in: {self.base_log_dir}")
        print(f"Applying Moving Average Smoothing with Window Size: {self.smoothing_window}")

        for alg_name in self.TAG_MAP.keys():
            alg_path = os.path.join(self.base_log_dir, alg_name)
            if not os.path.isdir(alg_path):
                print(f"Warning: Directory not found for algorithm: {alg_name}")
                continue

            # Use the default tag for the extraction, which is set to 'cum_rewards'
            metric_tag = self.DEFAULT_TAG
            run_count = 0

            # os.walk traverses the directory tree recursively
            for root, dirs, files in os.walk(alg_path):
                # The log_dir argument is now used as the starting point for recursive search within the _extract_data method
                try:
                    # Pass the root as the starting directory for the robust log finder
                    steps, values = self._extract_data(root, metric_tag)

                    # Check for duplicates or empty results before smoothing
                    if steps.size > 0:
                        # Apply Moving Average Smoothing HERE 🌟
                        smoothed_values = self._apply_moving_average(values, self.smoothing_window)

                        self.all_data[alg_name].append((steps, smoothed_values))
                        run_count += 1
                        print(f"  -> Found {alg_name} run: {os.path.basename(root)}")

                except KeyError as e:
                    # This handles missing tags gracefully (most common error)
                    # print(f"  -> Skipping path {root} for {alg_name}: {e}")
                    pass
                except Exception as e:
                    # This handles other unexpected errors (e.g., file corruption)
                    # print(f"  -> Skipping path {root} for {alg_name} due to unexpected error: {e}")
                    pass

            print(f"\n✅ Finished searching {alg_name}. Total runs found: {run_count}\n")

    def average_curves(self):
        """
        Averages the reward curves across multiple runs for each algorithm.
        This requires resampling all curves onto a common step base.
        """
        for alg_name, data_list in self.all_data.items():
            if not data_list:
                continue

            # 1. Determine the common maximum step (truncate to the shortest run if preferred)
            max_steps = max(steps.max() for steps, _ in data_list)

            # 2. Define a common step base for interpolation/resampling
            # We use 500 points for smooth averaging and plotting
            common_steps = np.linspace(0, max_steps, 500)

            # 3. Interpolate and collect values
            interpolated_values = []
            for steps, values in data_list:
                # Use numpy.interp for linear interpolation
                # Extrapolate beyond the last point with the last observed value (fill_value)
                # It's crucial to handle runs where the steps are identical to avoid errors
                if len(steps) > 1:
                    interp_values = np.interp(common_steps, steps, values, right=values[-1])
                    interpolated_values.append(interp_values)
                # else: print(f"Skipping run with insufficient data points for interpolation in {alg_name}")

            # Only proceed if we have valid interpolated data
            if interpolated_values:
                # 4. Calculate the mean curve
                mean_values = np.mean(np.array(interpolated_values), axis=0)
                self.averaged_curves[alg_name] = (common_steps, mean_values)

                print(f"📈 Averaged curve calculated for {alg_name} (runs used: {len(interpolated_values)}).")
            # else: print(f"No valid runs found for averaging {alg_name}.")

    def calculate_auc_for_averaged_curves(self):
        """ Calculates and prints the AUC for each averaged curve. """
        print("\n--- AUC Results (Averaged Curves) ---")
        auc_results = {}
        for alg_name, (steps, values) in self.averaged_curves.items():
            # AUC calculation using the trapezoidal rule
            auc_score = trapz(values, steps)
            auc_results[alg_name] = auc_score
            print(f"{alg_name.upper():<5}: {auc_score:,.2f}")

        # Sort results for easy comparison (highest AUC first)
        sorted_auc = sorted(auc_results.items(), key=lambda item: item[1], reverse=True)
        print("\n🏆 Ranked AUC Scores:")
        for alg, auc in sorted_auc:
            print(f"  {alg.upper():<5}: {auc:,.2f}")
        return auc_results

    def plot_averaged_curves(self, title="Averaged Learning Curves Comparison"):
        """ Plots all averaged learning curves for visual comparison. """

        if not self.averaged_curves:
            print("No averaged curves to plot.")
            return

        plt.figure(figsize=(12, 7))

        for alg_name, (steps, values) in self.averaged_curves.items():
            plt.plot(steps, values, label=alg_name.upper(), linewidth=2)

        plt.title(title, fontsize=16)
        plt.xlabel("Training Steps", fontsize=14)
        # Use the actual tag name for the plot label
        plt.ylabel(f"Averaged cumulative rewards", fontsize=14)
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(BASE_LOG_DIR + '/averaged_cum.png')
        plt.show()


# ----------------------------------------------------------------------
# Execution Block
# ----------------------------------------------------------------------

if __name__ == '__main__':
    # --------------------------------------------------------------------------------------
    # ⚠️ IMPORTANT: SET YOUR BASE DIRECTORY HERE
    # The script assumes the structure: BASE_DIR / algorithm_name / subfolder / tfevents_file
    # e.g., BASE_DIR/ppo/run_1/.../tfevents_file
    # --------------------------------------------------------------------------------------

    # Example adjustment (You MUST set this correctly based on your file system)
    # The path should point to the root folder containing the subdirectories for 'sac', 'td3', etc.
    BASE_LOG_DIR = '/home/ruben/Desktop/BM_Logs/comparisons/2025_11_18_variability_study/final/training_metrics_tensorboard'

    # --- Moving Average Configuration ---
    # Configure the window size here:
    SMOOTHING_WINDOW_SIZE = 100
    # ------------------------------------

    # --- Initialize and Run ---
    analyzer = AUCCalculator(BASE_LOG_DIR, smoothing_window=SMOOTHING_WINDOW_SIZE)

    # 1. Find logs and extract data
    analyzer.find_logs_and_extract()

    # 2. Average the curves
    analyzer.average_curves()

    # 3. Calculate AUC
    analyzer.calculate_auc_for_averaged_curves()

    # 4. Plot results
    analyzer.plot_averaged_curves()