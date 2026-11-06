"""Bounded, offline, same-pipeline experiments on selected video timestamps.

Random-seek samples are independent static probes, NOT a temporal-memory or
accuracy benchmark. No source frames/crops are persisted. Warmup is separate.
"""

import argparse
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import tempfile
import subprocess
from time import perf_counter

import av
from filelock import FileLock
import numpy as np
from PIL import Image
import torch

from uav_refmatch.config import load_config
from uav_refmatch.core.runtime import lock_path
from uav_refmatch.core.types import FrameRecord
from uav_refmatch.evaluation.profiler import Profiler
from uav_refmatch.outputs.recorder import Recorder, file_hash, write_json
from uav_refmatch.service import build_pipeline


def read_sample(video, timestamp):
    start = perf_counter()
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        origin = float(stream.start_time * stream.time_base) if stream.start_time else 0
        target = origin + timestamp
        container.seek(int(target / float(stream.time_base)), stream=stream)
        for frame in container.decode(stream):
            if frame.pts is not None and float(frame.pts * frame.time_base) >= target:
                image = frame.to_ndarray(format="rgb24")
                # A random seek has no trustworthy global decode index; never invent one.
                return FrameRecord(
                    0,
                    -1,
                    float(frame.pts * frame.time_base),
                    frame.width,
                    frame.height,
                    image,
                    (perf_counter() - start) * 1000,
                )
    raise ValueError(f"No source frame at/after timestamp {timestamp}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--config", type=Path, nargs="+", required=True)
    parser.add_argument("--timestamps", type=float, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.max_seconds <= 0 or any(t < 0 for t in args.timestamps):
        parser.error("Positive budget/repeats and nonnegative timestamps required")
    if args.output.exists():
        parser.error("Output already exists; choose a new result filename")
    os.environ["HF_HUB_OFFLINE"] = "1"
    np.random.seed(42)
    torch.manual_seed(42)
    result = dict(
        kind="independent_static_probes",
        source_index_policy="unknown=-1",
        source_sha256=file_hash(args.video),
        reference_sha256=file_hash(args.reference),
        requested_timestamps=args.timestamps,
        repeats=args.repeats,
        max_seconds=args.max_seconds,
        accuracy_metrics=None,
        variants=[],
    )
    root = Path(__file__).resolve().parents[1]
    result["source_hashes"] = {
        str(p.relative_to(root)): file_hash(p) for p in sorted((root / "src").rglob("*.py"))
    }
    result["git_commit"] = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    result["git_dirty"] = bool(
        subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True
        ).strip()
    )
    result["timing_scope"] = (
        "stage_timings_ms exclude model load, R0 encoding, random seek, "
        "resource sampling and manifest IO; probe_wall_seconds includes "
        "R0 encoding/recording, seek_decode_ms is reported separately"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(args.reference) as image:
        reference = np.array(image.convert("RGB"))
    start = perf_counter()
    with FileLock(str(lock_path()), timeout=0), tempfile.TemporaryDirectory() as temporary:
        try:
            for config_path in args.config:
                config = load_config(config_path, mode="static", warmup=False)
                variant = dict(config=config.model_dump(), cases=[], status="running")
                result["variants"].append(variant)
                tick = perf_counter()
                pipeline, provenance = build_pipeline(config)
                variant.update(models=provenance, load_seconds=perf_counter() - tick)
                # Same operations as inference, excluded from timed repetitions.
                warm = read_sample(args.video, args.timestamps[0])
                tick = perf_counter()
                pipeline.proposals.propose(warm.image)
                pipeline.encoder.encode([reference])
                pipeline.matcher.match(reference, reference)
                Profiler(pipeline.device).sync()
                variant["warmup_seconds"] = perf_counter() - tick
                del warm
                for timestamp in args.timestamps:
                    for repeat in range(args.repeats):
                        if perf_counter() - start >= args.max_seconds:
                            variant["status"] = "budget_exhausted"
                            return
                        frame = read_sample(args.video, timestamp)
                        recorder = Recorder(temporary, config, "probe")
                        state = {}
                        tick = perf_counter()
                        try:
                            status = pipeline.run(
                                [replace(frame, decode_ms=0)],
                                reference,
                                args.reference.stem,
                                recorder,
                                progress=state.update,
                            )
                        except Exception as exc:
                            recorder.finish("failed", type(exc).__name__)
                            raise
                        recorder.finish(status)
                        prediction = json.loads((recorder.path / "predictions.jsonl").read_text())
                        case = dict(
                            requested_timestamp=timestamp,
                            repeat=repeat,
                            seek_decode_ms=frame.decode_ms,
                            probe_wall_seconds=perf_counter() - tick,
                            prediction=prediction,
                            stage_timings_ms=state["stage_timings_ms"],
                        )
                        import csv

                        with (recorder.path / "resource_usage.csv").open() as handle:
                            case["resources"] = next(csv.DictReader(handle))
                        variant["cases"].append(case)
                        write_json(args.output, result)
                        print(
                            json.dumps(
                                dict(
                                    timestamp=timestamp,
                                    repeat=repeat,
                                    status=prediction["status"],
                                    candidates=prediction["candidate_count"],
                                    cosine=prediction["dino_cosine"],
                                    frame_ms=state["last_frame_ms"],
                                )
                            ),
                            flush=True,
                        )
                        del frame, recorder
                variant["status"] = "completed"
                del pipeline
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        except Exception as exc:
            if result["variants"]:
                result["variants"][-1].update(status="failed", error=type(exc).__name__)
            raise
        finally:
            result["wall_seconds"] = perf_counter() - start
            write_json(args.output, result)


if __name__ == "__main__":
    main()
