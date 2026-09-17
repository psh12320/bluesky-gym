import signal
import time

from stable_baselines3.common.callbacks import BaseCallback


class StopBeforeDeadline(BaseCallback):
    """Exit the learning loop cleanly so the trainer saves before a Slurm limit."""

    def __init__(self, max_seconds=0):
        super().__init__()
        self.max_seconds = max_seconds
        self.stop_requested = False

    def _on_training_start(self):
        self.started = time.monotonic()
        self.previous_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._request_stop)

    def _request_stop(self, signum, frame):
        self.stop_requested = True

    def _on_step(self):
        expired = self.max_seconds > 0 and time.monotonic() - self.started >= self.max_seconds
        if self.stop_requested or expired:
            print("Stopping training and saving the current model.", flush=True)
            return False
        return True

    def _on_training_end(self):
        signal.signal(signal.SIGTERM, self.previous_handler)
