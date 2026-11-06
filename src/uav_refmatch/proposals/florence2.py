import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from .boxes import clean_proposals
from ..core.runtime import resolve_revision


class Florence2Proposals:
    def __init__(self, model_config, proposal_config, device):
        self.config = proposal_config
        self.model_config = model_config
        self.device = device
        self.revision = resolve_revision(model_config.florence_id, model_config.florence_revision)
        self.dtype = torch.float16 if model_config.precision == "fp16" else torch.float32
        # Only the two allowlisted Microsoft repositories from validated config may execute code.
        self.processor = AutoProcessor.from_pretrained(
            model_config.florence_id, revision=self.revision, trust_remote_code=True
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_config.florence_id,
                revision=self.revision,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                attn_implementation="eager",
            )
            .eval()
            .to(device)
        )
        self.model.requires_grad_(False)

    def propose(self, image):
        inputs = self.processor(
            text="<REGION_PROPOSAL>", images=Image.fromarray(image), return_tensors="pt"
        )
        inputs = {
            k: v.to(self.device, dtype=self.dtype if v.is_floating_point() else v.dtype)
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            ids = self.model.generate(
                **inputs,
                max_new_tokens=self.model_config.max_new_tokens,
                do_sample=False,
                num_beams=3,
                use_cache=False,
            )
        text = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            text, task="<REGION_PROPOSAL>", image_size=(image.shape[1], image.shape[0])
        )
        boxes = parsed.get("<REGION_PROPOSAL>", {}).get("bboxes", [])
        return clean_proposals(
            boxes, image.shape[1], image.shape[0], self.config, self.model_config.florence_id
        )
