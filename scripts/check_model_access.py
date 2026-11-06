"""Check actual upstream access without downloading multi-GB weights or printing credentials."""

import json
from huggingface_hub import HfApi, hf_hub_download

models = [
    "microsoft/Florence-2-large-ft",
    "microsoft/Florence-2-base-ft",
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "facebook/dinov3-vitl16-pretrain-sat493m",
]
results = {}
for model in models:
    try:
        info = HfApi().model_info(model)
        hf_hub_download(model, "config.json", revision=info.sha)
        results[model] = dict(revision=info.sha, config_access=True)
    except Exception as exc:
        results[model] = dict(
            config_access=False,
            error_type=type(exc).__name__,
            http_status=getattr(getattr(exc, "response", None), "status_code", None),
        )
print(json.dumps(results, indent=2))
