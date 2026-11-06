import torch
import numpy as np
from ..config import Models, Proposals
from ..core.types import Proposal
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from .boxes import clean_proposals
from ..core.runtime import resolve_revision
from .cache_compat import enable_legacy_cache


class Florence2Proposals:
    def __init__(self, model_config: Models, proposal_config: Proposals, device: str) -> None:
        self.config = proposal_config
        self.model_config = model_config
        self.device = device
        self.revision = resolve_revision(model_config.florence_id, model_config.florence_revision)
        precision = model_config.florence_precision or model_config.precision
        if precision == "fp16" and device == "cpu":
            raise ValueError("Florence fp16 is not validated on CPU; select fp32")
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        # Only the two allowlisted Microsoft repositories from validated config may execute code.
        self.processor = AutoProcessor.from_pretrained(
            model_config.florence_id, revision=self.revision, trust_remote_code=True
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_config.florence_id,
                revision=self.revision,
                trust_remote_code=True,
                dtype=self.dtype,
                attn_implementation="eager",
            )
            .eval()
            .to(device)
        )
        self.model.requires_grad_(False)
        if model_config.florence_use_cache:
            enable_legacy_cache(self.model)
        self.last_diagnostics = {}

    def propose(self, image: np.ndarray) -> list[Proposal]:
        self.last_diagnostics = {}
        inputs = self.processor(
            text="<REGION_PROPOSAL>",
            images=Image.fromarray(image),
            return_tensors="pt",
        )
        inputs = {
            k: v.to(self.device, dtype=self.dtype if v.is_floating_point() else v.dtype)
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            ids = self.model.generate(
                # Original Microsoft generate() merges image tokens internally.
                # Its text-only attention mask must not be passed to the merged encoder.
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.model_config.max_new_tokens,
                do_sample=False,
                num_beams=self.model_config.florence_num_beams,
                use_cache=self.model_config.florence_use_cache,
                early_stopping=self.model_config.florence_num_beams > 1,
            )
        text = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            text, task="<REGION_PROPOSAL>", image_size=(image.shape[1], image.shape[0])
        )
        boxes = parsed.get("<REGION_PROPOSAL>", {}).get("bboxes", [])
        proposals = clean_proposals(
            boxes, image.shape[1], image.shape[0], self.config, self.model_config.florence_id
        )
        eos = self.model.language_model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        self.last_diagnostics = dict(
            raw_boxes=boxes,
            raw_count=len(boxes),
            kept_count=len(proposals),
            generated_tokens=int(ids.shape[-1]) - 1,
            reached_eos=int(ids[0, -1]) in eos_ids,
            hit_token_limit=int(ids.shape[-1]) - 1 >= self.model_config.max_new_tokens,
            num_beams=self.model_config.florence_num_beams,
            use_cache=self.model_config.florence_use_cache,
            input_size=list(inputs["pixel_values"].shape[-2:]),
        )
        return proposals
