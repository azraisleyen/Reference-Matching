import json
from uav_refmatch.config import Config
from uav_refmatch.outputs.recorder import Recorder, write_json
from uav_refmatch.worker import ExperimentWorker


def test_abnormal_worker_exit_finalizes_manifest(tmp_path):
    recorder = Recorder(tmp_path, Config())
    recorder.finish("failed")
    manifest = json.loads((recorder.path / "run_manifest.json").read_text())
    manifest["status"] = "running"
    write_json(recorder.path / "run_manifest.json", manifest)
    worker = ExperimentWorker(tmp_path)
    worker.state = {"run_directory": str(recorder.path)}
    worker._finalize_interrupted("failed", "worker_exit_9")
    saved = json.loads((recorder.path / "run_manifest.json").read_text())
    assert saved["status"] == "failed" and saved["error"] == "worker_exit_9"
    assert saved["ended_at"]
    assert (recorder.path / "summary.json").is_file()
