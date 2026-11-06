"""Real-model smoke experiment on a temporary video; no accuracy claim.

Uses an explicitly supplied image to generate a 0.4-second video in the OS temp
area. Inputs and crops never enter the project. Models load once; static/dynamic
runs receive the same instances. This validates plumbing, not UAV performance.
"""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import av
import cv2
import numpy as np
from PIL import Image
from uav_refmatch.config import Config
from uav_refmatch.service import build_pipeline, run_experiment
from uav_refmatch.outputs.video_renderer import render_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--rotation-deg",
        type=float,
        default=0,
        help="Controlled synthetic view change, not benchmark data",
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    args = parser.parse_args()
    config = Config(warmup=False)
    pipeline, provenance = build_pipeline(config)

    def reuse(cfg):
        pipeline.config = cfg
        return pipeline, provenance

    results = []
    with tempfile.TemporaryDirectory(prefix="uav-real-model-smoke-") as directory:
        root = Path(directory)
        image = np.array(Image.open(args.image).convert("RGB").resize((640, 480)))
        reference = root / "reference.png"
        Image.fromarray(image).save(reference)
        video_image = image
        if args.rotation_deg:
            matrix = cv2.getRotationMatrix2D((320, 240), args.rotation_deg, 0.85)
            video_image = cv2.warpAffine(image, matrix, (640, 480), borderValue=(128, 128, 128))
        video = root / "synthetic_source.mp4"
        with av.open(str(video), "w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width, stream.height = 640, 480
            stream.pix_fmt = "yuv420p"
            for index in range(12):
                frame = av.VideoFrame.from_ndarray(video_image, format="rgb24")
                frame.pts, frame.time_base = index, Fraction(1, 30)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        for mode in ("static", "dynamic"):
            cfg = Config(mode=mode, warmup=False)
            run = run_experiment(
                video,
                reference,
                cfg,
                args.output_root,
                "real_model_smoke",
                pipeline_factory=reuse,
                progress=lambda state: print(json.dumps(state), flush=True),
            )
            rows = [json.loads(line) for line in (run / "predictions.jsonl").open()]
            assert len(rows) == 3 and all(row["status"] != "ERROR" for row in rows)
            rendered = render_video(run, video)
            assert rendered.is_file()
            results.append(
                {
                    "run": run.name,
                    "counts": json.loads((run / "summary.json").read_text())["counts"],
                }
            )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
