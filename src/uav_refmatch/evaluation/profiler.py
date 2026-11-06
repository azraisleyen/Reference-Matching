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
        result = dict(
            cpu_ram_bytes=psutil.Process().memory_info().rss,
            cuda_allocated_bytes=0,
            cuda_reserved_bytes=0,
            peak_gpu_bytes=0,
            **memory.usage(),
        )
        if self.device == "cuda":
            result.update(
                cuda_allocated_bytes=torch.cuda.memory_allocated(),
                cuda_reserved_bytes=torch.cuda.memory_reserved(),
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            )
        return result
