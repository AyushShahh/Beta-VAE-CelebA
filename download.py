from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ayushshah/beta-vae-capacity-annealing-celeba",
    repo_type="model",
    local_dir=".",
    ignore_patterns=["README.md", ".gitignore"],
    local_dir_use_symlinks=False
)