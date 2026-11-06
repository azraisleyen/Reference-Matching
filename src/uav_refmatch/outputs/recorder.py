import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid
import yaml


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


class Recorder:
    def __init__(self, root, config, name=""):
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "-", name)[:40]
        run_id = f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{config.mode}_{config.models.encoder_id.split('/')[-1]}_{uuid.uuid4().hex[:8]}"
        if safe_name:
            run_id += "_" + safe_name
        self.path = Path(root) / run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.manifest = dict(
            run_id=run_id,
            started_at=now(),
            ended_at=None,
            status="running",
            processed_frames=0,
            mode=config.mode,
            configuration=config.model_dump(mode="json"),
        )
        self.handles = {}
        self.writers = {}
        (self.path / "resolved_config.yaml").write_text(
            yaml.safe_dump(config.model_dump(), sort_keys=False)
        )
        for filename in ("predictions.jsonl", "memory_events.jsonl", "run.log"):
            self.handles[filename] = (self.path / filename).open("w", encoding="utf-8")
        self.save_manifest()

    def save_manifest(self):
        write_json(self.path / "run_manifest.json", self.manifest)

    def log(self, event, **data):
        self.jsonline("run.log", dict(time=now(), event=event, **data))

    def jsonline(self, name, data):
        handle = self.handles[name]
        handle.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()

    def csv(self, name, data):
        if name not in self.writers:
            self.handles[name] = (self.path / name).open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(self.handles[name], fieldnames=list(data))
            writer.writeheader()
            self.writers[name] = writer
        self.writers[name].writerow(data)
        self.handles[name].flush()

    def prediction(self, prediction):
        data = asdict(prediction)
        self.jsonline("predictions.jsonl", data)
        row = {
            k: v
            for k, v in data.items()
            if k
            not in (
                "bbox_xyxy",
                "geometry",
                "verified_pairs",
                "retrieval_candidates",
                "proposals",
                "proposal_diagnostics",
            )
        }
        row.update(
            dict(zip(("x_min", "y_min", "x_max", "y_max"), prediction.bbox_xyxy or [None] * 4))
        )
        from ..core.types import GeometryEvidence

        geometry = (
            asdict(prediction.geometry)
            if prediction.geometry
            else dict.fromkeys(asdict(GeometryEvidence()), None)
        )
        row.update({f"geometry_{key}": value for key, value in geometry.items()})
        self.csv("detections.csv", row)
        self.manifest["processed_frames"] += 1

    def events(self, events):
        for event in events:
            self.jsonline("memory_events.jsonl", asdict(event))

    def finish(self, status, error=None):
        self.manifest.update(status=status, ended_at=now(), error=error)
        self.save_manifest()
        for handle in self.handles.values():
            handle.close()
        # Empty tables still have a meaningful header for startup failures/cancellation.
        for name, header in [
            ("detections.csv", "frame_index,status"),
            ("frame_timings.csv", "frame_index,total_frame_ms"),
            ("resource_usage.csv", "frame_index,cpu_ram_bytes"),
        ]:
            path = self.path / name
            if not path.exists():
                path.write_text(header + "\n")
        from .reporting import report

        report(self.path)
