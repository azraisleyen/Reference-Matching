"""Read existing run records without loading models or decoding media."""

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics


def describe(values):
    values = sorted(values)
    if not values:
        return None
    return dict(
        min=values[0],
        mean=statistics.mean(values),
        median=statistics.median(values),
        p95=values[round(0.95 * (len(values) - 1))],
        max=values[-1],
    )


def audit(root):
    results = []
    for manifest in sorted(root.glob("*/run_manifest.json")):
        directory = manifest.parent
        m = json.loads(manifest.read_text())
        names = [
            "run_manifest.json",
            "predictions.jsonl",
            "frame_timings.csv",
            "resource_usage.csv",
            "memory_events.jsonl",
        ]
        predictions = [json.loads(line) for line in (directory / names[1]).read_text().splitlines()]
        with (directory / names[2]).open() as stream:
            timings = list(csv.DictReader(stream))
        with (directory / names[3]).open() as stream:
            resources = list(csv.DictReader(stream))
        events = [json.loads(line) for line in (directory / names[4]).read_text().splitlines()]
        if not (len(predictions) == len(timings) == len(resources) == m["processed_frames"]):
            raise ValueError(f"Inconsistent record counts: {directory.name}")
        for rows in [predictions, timings, resources]:
            if [int(row["frame_index"]) for row in rows] != list(range(len(rows))):
                raise ValueError(f"Nonsequential frame records: {directory.name}")
        timing_stats = (
            {
                key: dict(
                    **describe([float(row[key]) for row in timings]),
                    total_ms=sum(float(row[key]) for row in timings),
                )
                for key in timings[0]
                if key != "frame_index"
            }
            if timings
            else {}
        )
        result = dict(
            run=directory.name,
            status=m["status"],
            device=m.get("models", {}).get("device"),
            frames=len(predictions),
            wall_seconds=(
                datetime.fromisoformat(m["ended_at"]) - datetime.fromisoformat(m["started_at"])
            ).total_seconds()
            if m.get("ended_at")
            else None,
            last_video_timestamp=predictions[-1]["timestamp_sec"] if predictions else None,
            counts=dict(Counter(row["status"] for row in predictions)),
            reasons=dict(Counter(row["decision_reason"] for row in predictions)),
            candidate_count_distribution=dict(
                Counter(row["candidate_count"] for row in predictions)
            ),
            total_candidates=sum(row["candidate_count"] for row in predictions),
            best_cosine=describe(
                [row["dino_cosine"] for row in predictions if row["dino_cosine"] is not None]
            ),
            geometry_pairs=sum(len(row["verified_pairs"]) for row in predictions),
            timings_ms=timing_stats,
            resources={
                key: describe([float(row[key]) for row in resources if row[key] != ""])
                for key in resources[0]
                if key != "frame_index"
            }
            if resources
            else {},
            memory_events=dict(Counter(row["event_type"] for row in events)),
            files_bytes={p.name: p.stat().st_size for p in directory.iterdir() if p.is_file()},
            evidence_sha256={
                name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names
            },
            blocks=[
                dict(
                    first=i,
                    last=min(i + 50, len(predictions)) - 1,
                    mean_frame_ms=statistics.mean(
                        float(row["total_frame_ms"]) for row in timings[i : i + 50]
                    ),
                    mean_candidates=statistics.mean(
                        row["candidate_count"] for row in predictions[i : i + 50]
                    ),
                )
                for i in range(0, len(predictions), 50)
            ],
        )
        results.append(result)
    return dict(
        percentile_method="sorted[round(0.95*(n-1))]",
        resource_note="Per-frame RSS samples, not continuous peaks or total system RAM.",
        runs=results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.runs), ensure_ascii=False, indent=2) + "\n")
