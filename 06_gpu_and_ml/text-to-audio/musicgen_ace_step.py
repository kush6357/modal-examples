import subprocess

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .uv_pip_install(
        "huggingface_hub[hf_transfer]",
        "torch",
        "torchvision",
        "torchaudio",
        "git+https://github.com/ace-step/ACE-Step.git",
    )
)
CACHE_DIR = "/cache"
volume = modal.Volume.from_name("musicgen-cache", create_if_missing=True)

app = modal.App("example-musicgen")


@app.function(
    image=image,
    max_containers=1,
    gpu="H100!",
    volumes={CACHE_DIR: volume},
    timeout=60 * 10,
)
@modal.web_server(7865, startup_timeout=60)
def ui():
    subprocess.Popen(
        (
            "acestep "
            f"--checkpoint_path {CACHE_DIR} "
            "--server_name '0.0.0.0' "
            "--port 7865 "
            # "--device_id 0 "
            # "--share true "
            "--bf16 true"
        ),
        shell=True,
    )
