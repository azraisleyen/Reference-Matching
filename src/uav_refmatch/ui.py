"""Local-only Gradio client: computations are delegated to one subprocess."""

import atexit
import json
import os
from pathlib import Path
import shutil
import tempfile
import gradio as gr
from .config import Config, LVD, SAT
from .data.video_reader import probe_video
from .outputs.video_renderer import inspect_frame, render_video
from .worker import ExperimentWorker


def build_app(output_root="runs", config=None):
    base_config = config or Config()
    worker = ExperimentWorker(output_root)
    root = Path(output_root).resolve()

    def selected(upload, path):
        value = path.strip() if path and path.strip() else upload
        if not value or not Path(value).expanduser().is_file():
            raise gr.Error("Geçerli dosya yükleyin veya yerel dosya yolunu girin")
        return str(Path(value).expanduser().resolve())

    def metadata(upload, path):
        try:
            info = probe_video(selected(upload, path))
            duration = info["duration_sec"]
            info["estimated_sample_count"] = (
                int(
                    duration
                    * min(base_config.target_fps, info["average_fps"] or base_config.target_fps)
                )
                if duration
                else None
            )
            return info
        except Exception as exc:
            return {"error": str(exc)}

    def config_for(mode, encoder, device, limit):
        data = base_config.model_dump()
        data["mode"] = mode
        data["models"]["device"] = device
        data["max_frames"] = int(limit) if limit else None
        data["models"]["encoder_id"] = LVD if encoder == "ViT-B/16 LVD" else SAT
        # A revision belongs to exactly one repository.
        if data["models"]["encoder_id"] != base_config.models.encoder_id:
            data["models"]["encoder_revision"] = (
                "5931719e67bbdb9737e363e781fb0c67687896bc"
                if data["models"]["encoder_id"] == LVD
                else "f692fa42da72c6797b67cd73494a168d1120d3ee"
            )
        return Config.model_validate(data)

    def start(v, vp, r, rp, mode, encoder, device, limit, name):
        try:
            return worker.start(
                selected(v, vp), selected(r, rp), config_for(mode, encoder, device, limit), name
            )
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def poll_status():
        state = worker.poll()
        expected = state.get("estimated_frames", 0)
        percent = min(99.9, 100 * state.get("processed_frames", 0) / expected) if expected else 0
        if state.get("phase") == "completed":
            percent = 100
        return state, percent

    def runs():
        return sorted(
            [p.name for p in root.glob("*") if (p / "run_manifest.json").is_file()], reverse=True
        )

    def run_path(name):
        if not name or name not in runs():
            raise gr.Error("Kayıtlı bir deney seçin")
        return root / name

    def review(name, index, source, show_proposals):
        try:
            return inspect_frame(
                run_path(name), index, source or None, show_proposals=show_proposals
            )
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def render(name, source):
        try:
            return str(render_video(run_path(name), source or None))
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    initial_runs = runs()
    initial_summary = {}
    if initial_runs and (root / initial_runs[0] / "summary.json").is_file():
        initial_summary = json.loads((root / initial_runs[0] / "summary.json").read_text())

    with gr.Blocks(title="UAV Reference Matching", delete_cache=(3600, 86400)) as app:
        gr.Markdown(
            "# UAV Reference Matching\nFlorence-2 → DINOv3 → EfficientLoFTR → RANSAC · Akademik baseline"
        )
        with gr.Tab("Deney"):
            with gr.Row():
                with gr.Column():
                    video = gr.File(label="UAV video", file_types=["video"], type="filepath")
                    video_path = gr.Textbox(label="Veya yerel video yolu")
                    ref = gr.File(
                        label="Başlangıç referansı", file_types=["image"], type="filepath"
                    )
                    ref_path = gr.Textbox(label="Veya yerel referans yolu")
                with gr.Column():
                    mode = gr.Radio(
                        ["dynamic", "static"], value=base_config.mode, label="Reference Mode"
                    )
                    encoder = gr.Dropdown(
                        ["ViT-B/16 LVD", "ViT-L/16 SAT"],
                        value="ViT-B/16 LVD"
                        if base_config.models.encoder_id == LVD
                        else "ViT-L/16 SAT",
                        label="DINOv3 Encoder",
                    )
                    name = gr.Textbox(label="Deney adı (isteğe bağlı)")
                    device = gr.Dropdown(
                        ["auto", "cpu", "mps", "cuda"],
                        value=base_config.models.device,
                        label="İşlem cihazı",
                    )
                    limit = gr.Number(
                        value=base_config.max_frames or 0,
                        minimum=0,
                        precision=0,
                        label="Kare sınırı (0: tüm video)",
                    )
                    gr.Markdown(
                        "Mac GPU için **mps** seçin. **auto**, CUDA yoksa CPU kullanır. "
                        "Önce kısa deney yapın; 7,5 FPS örnekleme, işlem hızı değildir."
                    )
                    info = gr.JSON(label="Video metadata")
            with gr.Accordion("Çalıştırma parametreleri", open=False):
                params = gr.JSON(
                    value=base_config.model_dump(), label="Kalibre edilmemiş başlangıç eşikleri"
                )
            with gr.Row():
                start_button = gr.Button("Start Experiment", variant="primary")
                cancel_button = gr.Button("Cancel Experiment")
            progress_bar = gr.Slider(
                0, 100, value=0, interactive=False, label="İlerleme (metadata temelli tahmin, %)"
            )
            status = gr.JSON(label="İlerleme / sayaçlar / süreler")
            cancel_message = gr.Textbox(label="İptal durumu", interactive=False)
            start_button.click(
                start,
                [video, video_path, ref, ref_path, mode, encoder, device, limit, name],
                status,
            )
            cancel_button.click(worker.cancel, outputs=cancel_message, queue=False)
            video.change(metadata, [video, video_path], info)
            video_path.blur(metadata, [video, video_path], info)
            video_path.submit(metadata, [video, video_path], info)
            for component in (mode, encoder, device, limit):
                component.change(
                    lambda m, e, d, n: config_for(m, e, d, n).model_dump(),
                    [mode, encoder, device, limit],
                    params,
                )
            gr.Timer(1).tick(poll_status, outputs=[status, progress_bar], queue=False)
        with gr.Tab("Sonuç inceleme"):
            choice = gr.Dropdown(runs(), label="Kayıtlı deney")
            gr.Button("Deney listesini yenile").click(
                lambda: gr.update(choices=runs()), outputs=choice
            )
            source = gr.Textbox(
                label="Kaynak taşındıysa aynı videonun yeni yolu (SHA256 kontrol edilir)"
            )
            index = gr.Number(value=0, minimum=0, precision=0, label="İşlenmiş kare indeksi")
            preview = gr.Image(label="Kaynak kare ve kayıtlı bbox", format="png")
            details = gr.JSON(label="Tahmin / geometri / bellek olayları")
            show_proposals = gr.Checkbox(
                label="Aday kutularını ve benzerlik skorlarını göster", value=True
            )
            gr.Button("Kareyi incele").click(
                review, [choice, index, source, show_proposals], [preview, details]
            )
            output = gr.File(label="Üretilen sonuç videosu")
            gr.Button("Video Oluştur").click(render, [choice, source], output)
            summary = gr.JSON(value=initial_summary, label="Deney özeti")
            choice.change(
                lambda n: json.loads((run_path(n) / "summary.json").read_text())
                if (run_path(n) / "summary.json").exists()
                else {},
                choice,
                summary,
            )
    app.unload(worker.cancel)
    app._uav_worker = worker
    return app


def launch(output_root="runs", config=None, port=7860):
    # Dedicated owned temp directory, never a user-selected source directory.
    temp = tempfile.mkdtemp(prefix="uav-refmatch-gradio-")
    os.environ["GRADIO_TEMP_DIR"] = temp
    app = build_app(output_root, config)

    def cleanup():
        app._uav_worker.close()
        shutil.rmtree(temp, ignore_errors=True)

    atexit.register(cleanup)
    app.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        allowed_paths=[],
        show_error=True,
        inbrowser=False,
    )
