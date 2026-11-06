import importlib.metadata
import platform
import shutil
import subprocess
import torch
import psutil


def doctor():
    packages = {}
    for name in (
        "torch",
        "torchvision",
        "transformers",
        "numpy",
        "opencv-python-headless",
        "av",
        "gradio",
        "pydantic",
        "pillow",
        "kornia",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = dict(
        python=platform.python_version(),
        physical_ram_bytes=psutil.virtual_memory().total,
        os=platform.system(),
        architecture=platform.machine(),
        packages=packages,
        cuda_runtime=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        mps_available=torch.backends.mps.is_available(),
        ffmpeg=bool(shutil.which("ffmpeg")),
        ffprobe=bool(shutil.which("ffprobe")),
    )
    if result["ffmpeg"]:
        result["ffmpeg_version"] = subprocess.check_output(
            ["ffmpeg", "-version"], text=True
        ).splitlines()[0]
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
        result["vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
        try:
            result["nvidia_driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            result["nvidia_driver"] = None
    return result
