from contextlib import contextmanager
from time import perf_counter
import psutil
import torch

STAGES = (
    "decode_ms",
    "proposal_ms",
    "dinov3_ms",
    "retrieval_ms",
    "efficientloftr_ms",
    "ransac_ms",
    "decision_ms",
    "memory_ms",
    "total_frame_ms",
)


class Profiler:
    def __init__(self, device):
        self.device = device
        self.values = dict.fromkeys(STAGES, 0.0)

    def sync(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    @contextmanager
    def measure(self, key):
        self.sync()
        start = perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.values[key] = self.values.get(key, 0) + (perf_counter() - start) * 1000

    def resources(self, memory):
        # Optional OS telemetry must not turn valid inference into a failed run.
        try:
            swap_used = psutil.swap_memory().used
        except (OSError, psutil.Error):
            swap_used = None
        result = dict(
            cpu_ram_bytes=psutil.Process().memory_info().rss,
            cuda_allocated_bytes=None,
            cuda_reserved_bytes=None,
            peak_gpu_bytes=None,
            mps_allocated_bytes=None,
            mps_driver_bytes=None,
            system_available_ram_bytes=psutil.virtual_memory().available,
            system_swap_used_bytes=swap_used,
            **memory.usage(),
        )
        if self.device == "cuda":
            result.update(
                cuda_allocated_bytes=torch.cuda.memory_allocated(),
                cuda_reserved_bytes=torch.cuda.memory_reserved(),
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            )
        elif self.device == "mps":
            result.update(
                mps_allocated_bytes=torch.mps.current_allocated_memory(),
                mps_driver_bytes=torch.mps.driver_allocated_memory(),
            )
        return result
