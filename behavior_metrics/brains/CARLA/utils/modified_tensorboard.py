
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorboardX import SummaryWriter
from tabulate import tabulate


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.fps_step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.txWriter = SummaryWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

    def _write_fps(self, fps):
        with self.writer.as_default():
            tf.summary.scalar("fps", fps, step=self.fps_step)
            self.fps_step += 1
            self.writer.flush()

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def update_fps(self, fps):
        self._write_fps(fps)
    def update_actions(self, actions, index):
        with self.writer.as_default():
            tf.summary.histogram("actions_v", actions[0], step=index)
            tf.summary.histogram("actions_w", actions[1], step=index)
            self.writer.flush()

    def update_weights(self, weights_paramaters, index):
        with self.writer.as_default():
            for name, param in weights_paramaters:
                # Convert PyTorch tensor to NumPy array
                param_numpy = param.cpu().detach().numpy() if param.device.type == 'cuda' else param.detach().numpy()
                tf.summary.histogram(name, param_numpy, step=index)
                self.writer.flush()

    def update_hyperparams(self, params):
        # Convert the dictionary to a list of (key, value) pairs
        table_data = [[key, value] for key, value in params.items()]
        # Create a nicely formatted table
        table = tabulate(table_data, headers=['Key', 'Value'], tablefmt='pipe')

        with self.writer.as_default():
            tf.summary.experimental.set_step(0)
            tf.summary.text("HyperParams", table)
            self.writer.flush()
