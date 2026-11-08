import pytest
from pydantic import ValidationError
from uav_refmatch.config import Config, load_config
from uav_refmatch.cli import main


def test_config_unknown_and_thresholds(tmp_path):
    with pytest.raises(ValidationError):
        Config(unknown=True)
    with pytest.raises(ValidationError):
        Config(memory={"memory_min_cosine": 0.2})
    config_path = tmp_path / "static.yaml"
    config_path.write_text("mode: static\n")
    assert load_config(config_path).mode == "static"


def test_cli_doctor(capsys):
    assert main(["doctor"]) == 0
    assert "cuda_available" in capsys.readouterr().out


def test_ui_build(tmp_path):
    from uav_refmatch.ui import build_app

    app = build_app(tmp_path)
    assert app.config["mode"] == "blocks"


def test_ui_callback_json_transport(tmp_path):
    import asyncio
    from fastapi.encoders import jsonable_encoder
    from uav_refmatch.ui import build_app

    app = build_app(tmp_path)
    index = next(i for i, fn in app.fns.items() if fn.fn and fn.fn.__name__ == "poll_status")
    result = asyncio.run(app.process_api(index, [], state=None))
    assert jsonable_encoder(result["data"]) == [{"phase": "idle"}, 0]


def test_default_config_is_loaded_and_missing_file_is_explicit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="--config"):
        load_config()
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/default.yaml").write_text("mode: static\ntarget_fps: 2\n")
    assert load_config().mode == "static"
    assert load_config().target_fps == 2
    assert load_config(mode="dynamic").mode == "dynamic"
