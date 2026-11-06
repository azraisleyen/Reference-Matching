import multiprocessing as mp
from pathlib import Path
import queue
import threading
from .config import Config
from .service import run_experiment


def _worker(video, reference, config, root, name, cancel, updates):
    try:
        run_experiment(
            video, reference, Config.model_validate(config), root, name, cancel.is_set, updates.put
        )
    except Exception as exc:
        updates.put(dict(phase="failed", error=f"{type(exc).__name__}: {exc}"))


class ExperimentWorker:
    def __init__(self, output_root):
        self.output_root = Path(output_root).resolve()
        self.context = mp.get_context("spawn")
        self.process = None
        self.cancel_event = None
        self.queue = None
        self.state = {"phase": "idle"}
        self.mutex = threading.Lock()

    def start(self, video, reference, config, name=""):
        with self.mutex:
            if self.process and self.process.is_alive():
                raise RuntimeError("Bir deney zaten çalışıyor")
            self.cancel_event = self.context.Event()
            self.queue = self.context.Queue()
            self.state = {"phase": "starting", "processed_frames": 0}
            self.process = self.context.Process(
                target=_worker,
                args=(
                    video,
                    reference,
                    config.model_dump(),
                    str(self.output_root),
                    name,
                    self.cancel_event,
                    self.queue,
                ),
                daemon=False,
            )
            self.process.start()
            return self.state.copy()

    def poll(self):
        with self.mutex:
            if self.queue:
                while True:
                    try:
                        self.state.update(self.queue.get_nowait())
                    except queue.Empty:
                        break
            if (
                self.process
                and not self.process.is_alive()
                and self.process.exitcode not in (None, 0)
            ):
                self.state.update(phase="failed", error=f"Worker exited: {self.process.exitcode}")
            return self.state.copy()

    def cancel(self):
        if self.cancel_event:
            self.cancel_event.set()
        return "İptal istendi; devam eden model çağrısı bitince kayıtlar kapatılacak."
