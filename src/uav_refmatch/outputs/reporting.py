from collections import Counter
import csv
import html
import json
from pathlib import Path
from .recorder import write_json


def report(run):
    run = Path(run)
    counts = Counter()
    score_sum = score_count = 0
    for line in (run / "predictions.jsonl").open():
        p = json.loads(line)
        counts[p["status"]] += 1
        if p["dino_cosine"] is not None:
            score_sum += p["dino_cosine"]
            score_count += 1
    events = Counter(
        json.loads(line)["event_type"] for line in (run / "memory_events.jsonl").open()
    )
    totals, n = Counter(), 0
    timings = run / "frame_timings.csv"
    if timings.exists():
        for row in csv.DictReader(timings.open()):
            n += 1
            for key, value in row.items():
                if key.endswith("_ms"):
                    totals[key] += float(value)
    result = dict(
        processed_frames=sum(counts.values()),
        counts=dict(counts),
        memory_events=dict(events),
        mean_cosine=score_sum / score_count if score_count else None,
        mean_timings_ms={k: v / n for k, v in totals.items()} if n else {},
        accuracy_metrics=None,
        note="Ground truth yok; doğruluk metrikleri hesaplanmadı.",
    )
    write_json(run / "summary.json", result)
    (run / "summary.html").write_text(
        '<!doctype html><html lang="tr"><meta charset="utf-8">'
        "<title>UAV deney özeti</title><style>body{font:16px system-ui;max-width:960px;margin:3rem auto;background:#f6f8fa;color:#17253c}pre{white-space:pre-wrap;background:white;padding:2rem}</style>"
        "<h1>UAV Reference Matching • Deney özeti</h1><pre>"
        + html.escape(json.dumps(result, ensure_ascii=False, indent=2))
        + "</pre></html>"
    )
    return result
