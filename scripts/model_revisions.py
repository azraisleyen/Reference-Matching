from huggingface_hub import HfApi

for name in ("facebook/dinov3-vitb16-pretrain-lvd1689m", "facebook/dinov3-vitl16-pretrain-sat493m"):
    print(name, HfApi().model_info(name).sha)
