import json
from PIL import Image
import pytest
from uav_refmatch.config import Config, Models, SAT
from uav_refmatch.core.errors import safe_error
from uav_refmatch.core.runtime import lock_path
from uav_refmatch.geometry.verifier import AffineRansacVerifier
from uav_refmatch.pipeline import Pipeline
from uav_refmatch.service import run_experiment
from filelock import FileLock
from test_frame_sampler import make_video
from test_pipeline import Proposals, Encoder, Matcher


def test_model_id_pins_own_revision():
    assert Models(encoder_id=SAT).encoder_revision == "f692fa42da72c6797b67cd73494a168d1120d3ee"
    assert (
        Models(florence_id="microsoft/Florence-2-base-ft").florence_revision
        == "f6c1a25888ffc1d945ee8a1a77ac833c7303d46e"
    )


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_lifecycle_manifest(tmp_path, monkeypatch, texture, status):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    video, reference = tmp_path / "video.mp4", tmp_path / "reference.jpg"
    make_video(video, count=8)
    Image.fromarray(texture).save(reference)

    def factory(config):
        if status == "failed":
            raise RuntimeError("gated checkpoint unavailable")
        return Pipeline(
            config, Proposals(), Encoder(), Matcher(), AffineRansacVerifier(config.geometry)
        ), {"test_double": True}

    kwargs = dict(
        video=video,
        reference=reference,
        config=Config(warmup=False),
        output_root=tmp_path / "runs",
        pipeline_factory=factory,
        cancel=lambda: status == "cancelled",
    )
    if status == "failed":
        with pytest.raises(RuntimeError, match="gated"):
            run_experiment(**kwargs)
        run = next((tmp_path / "runs").iterdir())
    else:
        run = run_experiment(**kwargs)
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["status"] == status and manifest["ended_at"]
    assert manifest["processed_frames"] == (2 if status == "completed" else 0)
    assert str(tmp_path) not in (run / "run_manifest.json").read_text()
    assert (run / "summary.html").is_file()
    assert not list(run.glob("*.jpg"))


def test_single_experiment_lock(tmp_path, monkeypatch, texture):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    source = tmp_path / "source.mp4"
    ref = tmp_path / "ref.jpg"
    make_video(source)
    Image.fromarray(texture).save(ref)
    with FileLock(str(lock_path()), timeout=0):
        with pytest.raises(RuntimeError, match="Başka bir deney"):
            run_experiment(source, ref, Config(), tmp_path / "runs")


def test_redacted_persisted_error():
    error = safe_error(
        RuntimeError(
            "read /Users/private/data/file.mp4 token=hf_secret https://example.test/?token=x"
        )
    )
    assert "hf_secret" not in error and "/Users/private" not in error and "https://" not in error
