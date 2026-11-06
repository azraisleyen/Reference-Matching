"""Post-hoc rendering only: no model imports, no frame caches, original VFR timing."""

from fractions import Fraction
import csv
import json
from pathlib import Path
import av
import cv2
from .recorder import write_json


def draw_prediction(image, prediction, overlay=True):
    output = image.copy()
    if prediction and prediction["status"] == "FOUND" and prediction["bbox_xyxy"] is not None:
        x1, y1, x2, y2 = prediction["bbox_xyxy"]
        h, w = output.shape[:2]
        a, b = max(0, min(w - 1, round(x1))), max(0, min(h - 1, round(y1)))
        c, d = max(0, min(w - 1, round(x2))), max(0, min(h - 1, round(y2)))
        cv2.rectangle(output, (a, b), (c, d), (0, 255, 0), 2)
        if overlay:
            text = f"{prediction['matched_memory_id']} t={prediction['timestamp_sec']:.3f}"
            cv2.putText(
                output, text, (a, max(15, b - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
    return output


def draw_proposals(image, prediction):
    """Diagnostic orange proposals; never presented as accepted green detections."""
    output = image.copy()
    scores = {p["candidate_index"]: p["cosine"] for p in prediction.get("retrieval_candidates", [])}
    for p in prediction.get("proposals", []):
        x1, y1, x2, y2 = map(round, p["bbox_xyxy"])
        index = p["candidate_index"]
        score = scores.get(index)
        label = f"candidate {index}" + (f" cos={score:.3f}" if score is not None else "")
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.putText(
            output, label, (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1
        )
    return output


def inspect_frame(run, frame_index, video=None, show_proposals=False):
    from ..service import resolve_source

    source = resolve_source(run, video)
    prediction = None
    for line in (Path(run) / "predictions.jsonl").open():
        row = json.loads(line)
        if row["frame_index"] == int(frame_index):
            prediction = row
            break
    if prediction is None:
        raise IndexError("İşlenmiş kare indeksi bulunamadı")
    with av.open(str(source)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index == prediction["source_frame_index"]:
                events = []
                for line in (Path(run) / "memory_events.jsonl").open():
                    event = json.loads(line)
                    if event["frame_index"] == int(frame_index):
                        events.append(event)
                details = dict(prediction=prediction, memory_events=events)
                for filename in ("frame_timings.csv", "resource_usage.csv"):
                    with (Path(run) / filename).open() as handle:
                        details[filename.removesuffix(".csv")] = next(
                            (
                                row
                                for row in csv.DictReader(handle)
                                if int(row["frame_index"]) == int(frame_index)
                            ),
                            None,
                        )
                image = frame.to_ndarray(format="rgb24")
                if show_proposals:
                    image = draw_proposals(image, prediction)
                return draw_prediction(image, prediction), details
    raise ValueError("Kaynak kare bulunamadı")


def render_video(run, video=None, overlay=True):
    from ..service import resolve_source

    run = Path(run)
    source = resolve_source(run, video)
    manifest = json.loads((run / "run_manifest.json").read_text())
    if manifest["status"] == "running":
        raise ValueError("Video üretmek için deneyin bitmesini bekleyin")
    partial_run = (
        manifest["status"] != "completed" or manifest["configuration"].get("max_frames") is not None
    )
    last_processed = max(
        (json.loads(line)["source_frame_index"] for line in (run / "predictions.jsonl").open()),
        default=-1,
    )
    if last_processed < 0:
        raise ValueError("Render edilecek tahmin kaydı yok")
    output = run / "annotated_video.mp4"
    partial = run / "annotated_video.partial.mp4"
    rows = (json.loads(line) for line in (run / "predictions.jsonl").open())
    current = next(rows, None)
    try:
        with av.open(str(source)) as reader, av.open(str(partial), "w") as writer:
            source_stream = reader.streams.video[0]
            stream = writer.add_stream("libx264", rate=source_stream.average_rate or Fraction(30))
            stream.width, stream.height = source_stream.width, source_stream.height
            stream.pix_fmt = (
                "yuv420p" if stream.width % 2 == 0 and stream.height % 2 == 0 else "yuv444p"
            )
            stream.time_base = source_stream.time_base
            stream.codec_context.time_base = source_stream.time_base
            stream.options = {"crf": "18", "preset": "medium"}
            previous_pts = None
            for index, frame in enumerate(reader.decode(source_stream)):
                if partial_run and index > last_processed:
                    break
                if frame.pts is None or (previous_pts is not None and frame.pts <= previous_pts):
                    raise ValueError("Rendering requires strictly increasing source PTS")
                previous_pts = frame.pts
                while current and current["source_frame_index"] < index:
                    current = next(rows, None)
                prediction = current if current and current["source_frame_index"] == index else None
                # No interpolation/held boxes on unsampled frames: their target location is unknown.
                image = draw_prediction(frame.to_ndarray(format="rgb24"), prediction, overlay)
                annotated = av.VideoFrame.from_ndarray(image, format="rgb24")
                annotated.pts, annotated.time_base = frame.pts, frame.time_base
                annotated.duration = frame.duration
                for packet in stream.encode(annotated):
                    writer.mux(packet)
            for packet in stream.encode():
                writer.mux(packet)
        partial.replace(output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    write_json(
        run / "render_manifest.json",
        dict(
            source_sha256=manifest["source_video"]["sha256"],
            timing="source_pts",
            boxes="sampled_frames_only",
            audio="omitted",
            orientation="coded_unrotated",
            overlay=overlay,
            truncated_to_last_processed_frame=partial_run,
        ),
    )
    return output
