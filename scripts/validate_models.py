"""Real adapters on upstream example images; no fake fallbacks, results are diagnostics."""

import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image
from uav_refmatch.config import Config
from uav_refmatch.core.runtime import device_for
from uav_refmatch.core.errors import safe_error
from uav_refmatch.geometry.verifier import AffineRansacVerifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component", choices=["florence", "dinov3", "efficientloftr"], required=True
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = Config()
    device = device_for(config.models)
    image = np.array(Image.open(args.image).convert("RGB").resize((640, 480)))
    result = {"component": args.component, "device": device, "verified": False}
    start = perf_counter()
    try:
        if args.component == "florence":
            from uav_refmatch.proposals.florence2 import Florence2Proposals

            model = Florence2Proposals(config.models, config.proposals, device)
            result["load_sec"] = perf_counter() - start
            start = perf_counter()
            proposals = model.propose(image)
            result.update(proposal_count=len(proposals), revision=model.revision)
        elif args.component == "dinov3":
            from uav_refmatch.features.dinov3 import DinoV3Encoder

            model = DinoV3Encoder(config.models, device)
            result["load_sec"] = perf_counter() - start
            start = perf_counter()
            embeddings = model.encode([image])
            result.update(
                shape=list(embeddings.shape),
                norm=float(np.linalg.norm(embeddings[0])),
                revision=model.revision,
            )
        else:
            from uav_refmatch.matching.efficientloftr import EfficientLoFTRMatcher

            model = EfficientLoFTRMatcher(config.models, device)
            result["load_sec"] = perf_counter() - start
            start = perf_counter()
            matches = model.match(image, image)
            evidence = AffineRansacVerifier(config.geometry).verify(
                matches, image.shape, image.shape
            )
            from dataclasses import asdict

            result.update(
                matches=len(matches.confidence),
                geometry=asdict(evidence),
                commit=model.commit,
                checkpoint_sha256=model.checkpoint_sha256,
            )
        result.update(verified=True, inference_sec=perf_counter() - start)
    except Exception as exc:
        result["error"] = safe_error(exc)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
