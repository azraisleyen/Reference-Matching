import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from .normalization import normalize
from ..core.runtime import resolve_revision


class DinoV3Encoder:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.revision = resolve_revision(config.encoder_id, config.encoder_revision)
        self.dtype = torch.float16 if config.precision == "fp16" else torch.float32
        # Processor and model always come from the same checkpoint/revision (including SAT).
        self.processor = AutoImageProcessor.from_pretrained(
            config.encoder_id, revision=self.revision
        )
        self.model = (
            AutoModel.from_pretrained(config.encoder_id, revision=self.revision, dtype=self.dtype)
            .eval()
            .to(device)
        )
        self.model.requires_grad_(False)
        self.dimension = self.model.config.hidden_size

    def encode(self, images):
        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)
        batches = []
        with torch.inference_mode():
            for start in range(0, len(images), self.config.batch_size):
                inputs = self.processor(
                    images=[
                        Image.fromarray(i) for i in images[start : start + self.config.batch_size]
                    ],
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device, dtype=self.dtype) for k, v in inputs.items()}
                cls = self.model(**inputs).last_hidden_state[:, 0, :].float().cpu().numpy()
                if cls.shape[1] != self.dimension:
                    raise ValueError("Unexpected DINOv3 embedding dimension")
                batches.append(normalize(cls))
        return np.concatenate(batches)
