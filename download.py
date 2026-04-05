from huggingface_hub import snapshot_download
import os
import shutil

snapshot_download(
    repo_id="ayushshah/beta-vae-capacity-annealing-celeba",
    repo_type="model",
    local_dir="./huggingface",
    local_dir_use_symlinks=False
)

os.makedirs("checkpoints", exist_ok=True)

shutil.move("./huggingface/directions", "./")
shutil.move("./huggingface/model.safetensors", "./checkpoints/model.safetensors")

shutil.rmtree("./huggingface")