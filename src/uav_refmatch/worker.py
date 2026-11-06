import json
import multiprocessing as mp
from pathlib import Path
import queue
import threading
from .config import Config
from .service import run_experiment
from .outputs.recorder import now, write_json
from .outputs.reporting import report


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
                self._finalize_interrupted("failed", self.state["error"])
            return self.state.copy()

    def cancel(self):
        if self.cancel_event:
            self.cancel_event.set()
        return "İptal istendi; devam eden model çağrısı bitince kayıtlar kapatılacak."

    def _finalize_interrupted(self, status, reason):
        value = self.state.get("run_directory")
        if not value:
            return
        path = Path(value).resolve()
        if path.parent != self.output_root:
            return
        manifest_path = path / "run_manifest.json"
        if not manifest_path.is_file():
            return
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] != "running":
            return
        manifest.update(status=status, ended_at=now(), error=reason)
        write_json(manifest_path, manifest)
        with (path / "run.log").open("a") as handle:
            handle.write(
                json.dumps(dict(time=now(), event="worker_interrupted", reason=reason)) + "\n"
            )
        report(path)

    def close(self):
        """Stop the owned worker before deleting owned Gradio temporary uploads."""
        self.cancel()
        if self.process and self.process.is_alive():
            self.process.join(timeout=30)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
                self._finalize_interrupted("cancelled", "forced_shutdown_after_cancel_timeout")
