"""Causal downstream ablation using recorded Florence boxes, never ground truth.

No Florence inference is repeated. All DINO/LoFTR/RANSAC and memory operations
are real. Missing intervals are NOT a complete-video memory evaluation.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from time import perf_counter

import av
from filelock import FileLock
import numpy as np
from PIL import Image

from uav_refmatch.config import load_config
from uav_refmatch.core.runtime import lock_path
from uav_refmatch.core.types import FrameRecord, Proposal
from uav_refmatch.features.dinov3 import DinoV3Encoder
from uav_refmatch.matching.efficientloftr import EfficientLoFTRMatcher
from uav_refmatch.geometry.verifier import AffineRansacVerifier
from uav_refmatch.pipeline import Pipeline
from uav_refmatch.outputs.recorder import Recorder, file_hash, write_json


class RecordedProposals:
    def __init__(self):
        self.current = None
        self.last_diagnostics = {}

    def propose(self, image):
        row = self.current
        self.last_diagnostics = {**row["proposal_diagnostics"], "replayed_from_record": True}
        return [
            Proposal(p["proposal_id"], tuple(p["bbox_xyxy"]), p["source_model"])
            for p in row["proposals"]
        ]


def frames(video, selected, adapter):
    # Authenticate source hash before this function. Seek only across omitted
    # intervals; preserve recorded indices and independently verify every PTS.
    groups = []
    for row in sorted(selected.values(), key=lambda r: r["frame_index"]):
        if not groups or row["frame_index"] != groups[-1][-1]["frame_index"] + 1:
            groups.append([])
        groups[-1].append(row)
    with av.open(str(video)) as container:
        for group in groups:
            container.seek(int(group[0]["timestamp_sec"] * 1_000_000))
            index = 0
            for frame in container.decode(video=0):
                row = group[index]
                if frame.time < row["timestamp_sec"] - 1e-6:
                    continue
                if abs(frame.time - row["timestamp_sec"]) > 1e-6:
                    raise ValueError("Source PTS mismatch")
                adapter.current = row
                yield FrameRecord(
                    row["frame_index"],
                    row["source_frame_index"],
                    frame.time,
                    frame.width,
                    frame.height,
                    frame.to_ndarray(format="rgb24"),
                )
                index += 1
                if index == len(group):
                    break
            if index != len(group):
                raise ValueError("Recorded window extends beyond source")


def read_csv(path):
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--config", type=Path, nargs="+", required=True)
    parser.add_argument("--windows", nargs="+", required=True, help="start:end seconds")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=180)
    args = parser.parse_args()
    if args.output.exists() or args.max_seconds <= 0:
        parser.error("Use a new output directory and positive budget")
    windows = [tuple(map(float, w.split(":"))) for w in args.windows]
    if any(len(w) != 2 or not 0 <= w[0] < w[1] for w in windows):
        parser.error("Invalid time windows")
    manifest = json.loads((args.run / "run_manifest.json").read_text())
    if (
        file_hash(args.video) != manifest["source_video"]["sha256"]
        or file_hash(args.reference) != manifest["reference_image"]["sha256"]
    ):
        raise ValueError("Source/reference hash mismatch")
    rows = [json.loads(line) for line in (args.run / "predictions.jsonl").read_text().splitlines()]
    selected = {
        r["source_frame_index"]: r
        for r in rows
        if any(a <= r["timestamp_sec"] < b for a, b in windows)
    }
    if not selected:
        raise ValueError("No recorded frames in windows")
    configs = [load_config(p, warmup=False) for p in args.config]
    original = load_config(args.run / "resolved_config.yaml", warmup=False)
    for c in configs:
        if c.proposals != original.proposals or c.models != original.models:
            raise ValueError("Replay requires identical proposal and model configuration")
    os.environ["HF_HUB_OFFLINE"] = "1"
    args.output.mkdir(parents=True)
    result = dict(
        script_sha256=file_hash(Path(__file__)),
        kind="causal_cached_proposal_windows",
        source_run=manifest["run_id"],
        source_sha256=manifest["source_video"]["sha256"],
        reference_sha256=manifest["reference_image"]["sha256"],
        windows=windows,
        requested_frames=len(selected),
        variants=[],
        limitation="Skipped intervals; downstream timing only, Florence generation omitted",
        source_hashes={
            str(p.relative_to(Path(__file__).resolve().parents[1])): file_hash(p)
            for p in (Path(__file__).resolve().parents[1] / "src").rglob("*.py")
        },
    )
    with Image.open(args.reference) as im:
        reference = np.array(im.convert("RGB"))
    start = perf_counter()
    with FileLock(str(lock_path()), timeout=0):
        encoder = DinoV3Encoder(configs[0].models, configs[0].models.device)
        matcher = EfficientLoFTRMatcher(configs[0].models, configs[0].models.device)
        encoder.encode([reference])
        matcher.match(reference, reference)
        try:
            for c in configs:
                if perf_counter() - start >= args.max_seconds:
                    break
                adapter = RecordedProposals()
                engine = Pipeline(
                    c,
                    adapter,
                    encoder,
                    matcher,
                    AffineRansacVerifier(c.geometry, c.seed),
                    c.models.device,
                )
                recorder = Recorder(args.output, c, "memory-probe")
                status = "failed"
                try:
                    status = engine.run(
                        frames(args.video, selected, adapter),
                        reference,
                        args.reference.stem,
                        recorder,
                        cancel=lambda: perf_counter() - start >= args.max_seconds,
                    )
                finally:
                    recorder.finish(status)
                predictions = [
                    json.loads(line)
                    for line in (recorder.path / "predictions.jsonl").read_text().splitlines()
                ]
                timings = read_csv(recorder.path / "frame_timings.csv")
                events = [
                    json.loads(line)
                    for line in (recorder.path / "memory_events.jsonl").read_text().splitlines()
                ]
                resources = read_csv(recorder.path / "resource_usage.csv")
                result["variants"].append(
                    dict(
                        config=c.model_dump(),
                        status=status,
                        predictions=predictions,
                        timings=timings,
                        events=events,
                        resources=resources,
                    )
                )
                write_json(args.output / "comparison.json", result)
                print(
                    json.dumps(
                        dict(
                            mode=c.mode,
                            status=status,
                            frames=len(predictions),
                            found=sum(p["status"] == "FOUND" for p in predictions),
                            accepted=sum(e["event_type"] == "ACCEPTED" for e in events),
                            max_refs=max(
                                (p["active_reference_count"] for p in predictions), default=0
                            ),
                        )
                    ),
                    flush=True,
                )
        finally:
            result["wall_seconds"] = perf_counter() - start
            write_json(args.output / "comparison.json", result)


if __name__ == "__main__":
    main()
