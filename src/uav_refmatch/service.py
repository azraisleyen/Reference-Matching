"""Experiment lifecycle shared by CLI and the UI's separate worker process."""

import json
from itertools import islice
from pathlib import Path
import random
import subprocess
from time import perf_counter
import numpy as np
from PIL import Image
import torch
from filelock import FileLock, Timeout
from .core.errors import safe_error
from .evaluation.profiler import Profiler
from .core.runtime import device_for, lock_path
from .data.video_reader import VideoReader, probe_video
from .doctor import doctor
from .outputs.recorder import Recorder, file_hash, write_json
from .pipeline import Pipeline


def source_registry(run):
    from hashlib import sha256

    root = lock_path().parent / "sources"
    root.mkdir(exist_ok=True)
    return root / (sha256(str(Path(run).resolve()).encode()).hexdigest() + ".json")


def resolve_source(run, video=None):
    if video:
        path = Path(video).expanduser().resolve()
    else:
        registry = source_registry(run)
        if not registry.exists():
            raise ValueError("Kaynak videoyu --video ile yeniden seçin")
        path = Path(json.loads(registry.read_text())["video"])
    if not path.is_file():
        raise FileNotFoundError("Kaynak video taşınmış; aynı videoyu yeniden seçin")
    manifest = json.loads((Path(run) / "run_manifest.json").read_text())
    if file_hash(path) != manifest["source_video"]["sha256"]:
        raise ValueError("Seçilen video deneyde kullanılan videoyla eşleşmiyor (SHA256)")
    return path


def build_pipeline(config):
    from .features.dinov3 import DinoV3Encoder
    from .proposals.florence2 import Florence2Proposals
    from .matching.efficientloftr import EfficientLoFTRMatcher
    from .geometry.verifier import AffineRansacVerifier

    device = device_for(config.models)
    # Fail early on gated DINO weights before loading the larger proposal model.
    encoder = DinoV3Encoder(config.models, device)
    matcher = EfficientLoFTRMatcher(config.models, device)
    proposals = Florence2Proposals(config.models, config.proposals, device)
    pipeline = Pipeline(
        config,
        proposals,
        encoder,
        matcher,
        AffineRansacVerifier(config.geometry, config.seed),
        device,
    )
    provenance = dict(
        florence_id=config.models.florence_id,
        florence_revision=proposals.revision,
        encoder_id=config.models.encoder_id,
        encoder_revision=encoder.revision,
        eloftr_commit=matcher.commit,
        eloftr_checkpoint_sha256=matcher.checkpoint_sha256,
        device=device,
        precision=config.models.precision,
        florence_precision=config.models.florence_precision or config.models.precision,
    )
    return pipeline, provenance


def run_experiment(
    video,
    reference,
    config,
    output_root="runs",
    name="",
    cancel=lambda: False,
    progress=lambda p: None,
    pipeline_factory=build_pipeline,
):
    video, reference = Path(video).expanduser().resolve(), Path(reference).expanduser().resolve()
    if not video.is_file() or not reference.is_file():
        raise FileNotFoundError("Video ve referans dosyaları bulunamadı")
    try:
        with FileLock(str(lock_path()), timeout=0):
            recorder = Recorder(output_root, config, name)
            progress(dict(run_directory=str(recorder.path), phase="initializing"))
            reader = None
            status, error = "failed", None
            try:
                random.seed(config.seed)
                np.random.seed(config.seed)
                torch.manual_seed(config.seed)
                metadata = probe_video(video)
                recorder.manifest.update(
                    environment=doctor(),
                    video_metadata=metadata,
                    source_video=dict(name=video.name, sha256=file_hash(video)),
                    reference_image=dict(name=reference.name, sha256=file_hash(reference)),
                    source_below_target_fps=bool(
                        metadata["average_fps"] and metadata["average_fps"] < config.target_fps
                    ),
                )
                try:
                    recorder.manifest["git_commit"] = subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
                    ).strip()
                except (FileNotFoundError, subprocess.CalledProcessError):
                    recorder.manifest["git_commit"] = None
                if recorder.manifest["git_commit"]:
                    recorder.manifest["git_dirty"] = bool(
                        subprocess.check_output(
                            ["git", "status", "--porcelain", "--untracked-files=no"], text=True
                        ).strip()
                    )
                registry = source_registry(recorder.path)
                write_json(registry, dict(video=str(video)))
                registry.chmod(0o600)
                recorder.save_manifest()
                if metadata["duration_sec"]:
                    import math

                    expected = math.ceil(
                        metadata["duration_sec"]
                        * min(config.target_fps, metadata["average_fps"] or config.target_fps)
                    )
                    progress(
                        dict(
                            estimated_frames=min(expected, config.max_frames)
                            if config.max_frames
                            else expected
                        )
                    )
                if cancel():
                    status = "cancelled"
                    return recorder.path
                progress(dict(phase="loading_models"))
                start = perf_counter()
                pipeline, provenance = pipeline_factory(config)
                recorder.manifest.update(
                    models=provenance, model_load_ms=(perf_counter() - start) * 1000
                )
                progress(dict(device=pipeline.device))
                with Image.open(reference) as image:
                    reference_image = np.array(image.convert("RGB"))
                timer = perf_counter()
                prepare = getattr(pipeline, "prepare_reference", None)
                recorder.manifest["proposal_reference_context"] = (
                    prepare(reference_image) if prepare else {}
                )
                Profiler(pipeline.device).sync()
                recorder.manifest["proposal_reference_prepare_ms"] = (perf_counter() - timer) * 1000
                recorder.manifest["warmup_ms"] = 0.0
                if config.warmup and not cancel():
                    progress(dict(phase="warmup"))
                    # Do not encode R0 twice: warm up using the first actual video frame.
                    warm_reader = iter(
                        VideoReader(video, config.target_fps, config.verified_cfr_fps)
                    )
                    try:
                        first = next(warm_reader)
                        timer = perf_counter()
                        pipeline.proposals.propose(first.image)
                        pipeline.encoder.encode([first.image])
                        pipeline.matcher.match(first.image, first.image)
                        Profiler(pipeline.device).sync()
                        recorder.manifest["warmup_ms"] = (perf_counter() - timer) * 1000
                    finally:
                        warm_reader.close()
                if pipeline.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                reader = VideoReader(video, config.target_fps, config.verified_cfr_fps)
                progress(dict(phase="inference"))
                status = pipeline.run(
                    islice(reader, config.max_frames) if config.max_frames else reader,
                    reference_image,
                    reference.stem,
                    recorder,
                    cancel,
                    progress,
                )
            except Exception as exc:
                error = safe_error(exc)
                detail = f"{type(exc).__name__}: {str(exc)}"
                # User sees actionable detail; persisted logs omit arbitrary paths/tokens from upstream exceptions.
                recorder.log("run_failed", error=error)
                progress(dict(phase="failed", error=detail))
                raise
            finally:
                if reader:
                    recorder.manifest.update(
                        timestamp_fallback_used=reader.fallback_used,
                        timestamp_fallback_frames=reader.fallback_frames,
                    )
                recorder.finish(status, error)
                progress(dict(run_directory=str(recorder.path), phase=status))
            return recorder.path
    except Timeout as exc:
        raise RuntimeError("Başka bir deney çalışıyor; önce tamamlayın veya iptal edin") from exc
