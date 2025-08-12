# ---
# cmd: ["modal", "run", "06_gpu_and_ml/llm-serving/triton_inference.py"] # ??
# ---

# # GPU Packing with the Triton Inference Server
# The [NVIDIA Triton](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html)
# inference server is an open-source, long-standing, and production-grade
# inference server. In particular, it boasts state of the art performance
# in GPU packing: serving multiple models from a single GPU. It has several
# backend options (TensorRT, PyTorch, vLLM, ONNX, [and more](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quick_deployment_by_backend.html)...)
#
# This guide shows how to set up a Triton server sitting behind
# a web endpoint on Modal. The only requirement is that you have
# fastapi in your local Python environment, as well as modal.
#
# # TODO:
# 1. get advice on how to do without local fastAPI??
# 2. add vllm backend, others?
#
# ## Local env imports
# # Import everything we need for the locally-run Python
import io
import json
import subprocess
import time
from pathlib import Path
from urllib.request import Request as URLRequest, urlopen

import modal
from fastapi import Body as FastAPIBody

# ## Constants
# The [Triton configuration schema](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)
# is comprehensive, but complicated. Instead of passing tons of arguments
# to our parameterized function, we'll define them with some constants
# here. You can also write your config by hand, and simply point the app
# to it when you deploy (keep reading for those instructions).

# Model Info
MODEL_NAME = "openai/clip-vit-base-patch16"
INPUT_SHAPE = [3, 224, 224]
OUTPUT_DIM = 768
IN_NAME = "clip_input"
OUT_NAME = "clip_output"

# Inference Server
N_INSTANCES = 2
REPO_PATH = None  # put a path to your model repo if you have one already!
TRITON_BACKEND = "pytorch"  # "tensorrt"
FORCE_REBUILD = False

# Must be 1 for TRT backend (will get overwritten)
MAX_BATCH_SIZE = 1
INPUT_NAME = "image_input"
OUTPUT_NAME = "embedding_output"

# Environment
GPU_CONFIG = "H100:1"

# ### Storage
# We use [Modal Volumes](https://modal.com/docs/guide/volumes#volumes)
# to create persistent storage for caching model weights, Triton
# configs, etc.
model_volume = modal.Volume.from_name("example-model-repo", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("huggingface-secret")
VOL_MNT = Path("/data")
HF_HOME = VOL_MNT / "huggingface"
MODEL_REPO = VOL_MNT / "triton_repo"

# # Image
# NVIDIA provides a convenient base container with the
# heavy duty backend stuff built-in. Since Modal lazily loads
# cached modules, we don't need to use a backend-specific
# image to make it as slim as possible.
TRITON_IMAGE = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/tritonserver:25.06-py3",
        add_python="3.12",
    )
    .uv_pip_install(
        "torch",
        "torchvision",
        "transformers",
        "pillow",
        "tritonclient[all]",
        "tqdm",
        "hf_transfer",
        "tensorrt",
        "onnx",
        "pynvml",
        "fastapi",
    )
    .env(
        {
            # For fast HuggingFace downloading+caching
            "HF_HOME": HF_HOME.as_posix(),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .entrypoint([])
)

# All functions and classes that belong to this app
# will inherit the image, volumes, and secrets designated here.
app = modal.App(
    "example-triton-server",
    image=TRITON_IMAGE,
    volumes={VOL_MNT: model_volume},
    secrets=[HF_SECRET],
)

with TRITON_IMAGE.imports():
    import numpy as np
    import tritonclient.grpc as grpcclient
    from PIL import Image
    from tqdm import tqdm
    from transformers import CLIPVisionModel

# # Triton Repo Building Helpers
# This section of the example shows how to programmatically
# build a Triton repo config file for a CLIPVisionModel. You will
# need to customize this for your model type.
#
# `build_triton_repo` function compiles or otherwise creates the necessary Triton
# artifacts needed for Triton to recognize the model.


def build_triton_repo(
    triton_model_name: str,
    version: str = "1",
):
    """
    Build (or reuse) a Triton model repository for a CLIP vision encoder.

    Parameters
    ----------
    triton_model_name : str
        Directory/name visible to Triton. Must match `config.pbtxt:name`.
    TRITON_BACKEND : {"pytorch","tensorrt"}
        - "pytorch": saves TorchScript to `model.pt` (backend: pytorch).
        - "tensorrt": exports ONNX then builds `model.plan` (backend: tensorrt).
    version : str
        Triton model version subdirectory (e.g., "1").
    FORCE_REBUILD : bool
        If True, ignore existing artifacts and rebuild.

    Side effects
    ------------
    - Writes `config.pbtxt` and the backend artifact into the Modal Volume.
    - Calls `model_volume.commit()` so later containers can reuse the repo.

    Notes
    -----
    - IO tensor names (`IN_NAME`, `OUT_NAME`) must match `config.pbtxt`.
    - If you change input shape, regenerate both the artifact and config.
    """

    import torch
    from torch.onnx import export as onnx_export

    # 0. short-circuit if artifacts & config already exist
    repo_dir = Path(MODEL_REPO) / triton_model_name / version
    repo_dir.mkdir(parents=True, exist_ok=True)
    max_batch_size = 1 if TRITON_BACKEND == "tensorrt" else MAX_BATCH_SIZE

    artifact = repo_dir / ("model.pt" if TRITON_BACKEND == "pytorch" else "model.plan")
    cfg_file = Path(MODEL_REPO) / triton_model_name / "config.pbtxt"
    if artifact.exists() and cfg_file.exists() and (not FORCE_REBUILD):
        print("Model repo already complete - skip build.")
        return

    # 1. Define model in PyTorch (used for both backends)
    st = time.perf_counter()

    class ClipEmbedder(torch.nn.Module):
        def __init__(self, hf_name: str):
            super().__init__()
            self.clip = CLIPVisionModel.from_pretrained(hf_name)
            self.clip.half()
            self.clip.eval()

        @torch.no_grad()
        def forward(self, pixels: torch.Tensor):
            return self.clip(pixel_values=pixels).pooler_output

    model = ClipEmbedder(MODEL_NAME).eval()
    example = torch.randn(
        (max_batch_size, *INPUT_SHAPE),
        device="cuda",
        dtype=torch.float16,
    )
    print(f"Building model took {time.perf_counter() - st:.2E}s")

    # 2.  Write backend-specific artifact
    if TRITON_BACKEND == "pytorch":
        print("doing torch trace...", end="")
        st = time.perf_counter()
        model = model.cuda()
        traced = torch.jit.trace(model, example, strict=False).cpu()
        # rename io so we have input0 / output0
        graph = traced.inlined_graph
        g_inputs, g_outputs = list(graph.inputs()), list(graph.outputs())
        g_inputs[0].setDebugName(IN_NAME)
        g_outputs[0].setDebugName(OUT_NAME)
        traced.save(artifact)
        # Free GPU memory
        del model, traced
        torch.cuda.empty_cache()
        print(f"took {time.perf_counter() - st:.2E}s")

    elif TRITON_BACKEND == "tensorrt":
        onnx_path = repo_dir / "model.onnx"
        st = time.perf_counter()
        if not onnx_path.is_file():
            onnx_export(
                model.cpu(),  # ONNX must be on CPU
                example.cpu(),
                onnx_path,
                input_names=[IN_NAME],
                output_names=[OUT_NAME],
                dynamic_axes={IN_NAME: {0: "batch"}, OUT_NAME: {0: "batch"}},
                opset_version=17,
            )
        print(f"\n\tPyTorch->ONNX conversion took {time.perf_counter() - st:.2E}s")

        bsz_str = f"{max_batch_size}x"
        inp_str = "x".join(str(d) for d in INPUT_SHAPE)

        st = time.perf_counter()
        plan_path = repo_dir / "model.plan"
        # --fp16 flag assumes GPU supports it; change to --fp32 if not
        cmd = [
            "/usr/src/tensorrt/bin/trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={plan_path}",
            f"--fp16--minShapes={IN_NAME}:1x{inp_str}",
            f"--optShapes={IN_NAME}:{bsz_str}{inp_str}",
            f"--maxShapes={IN_NAME}:{bsz_str}{inp_str}",
            "--verbose",
        ]

        subprocess.run(cmd, check=True)
        print(f"\n\tONNX->TRT conversion took {time.perf_counter() - st:.2E}s")

    else:
        raise ValueError(
            f"Triton backend `{TRITON_BACKEND}` not"
            "recognized; try `pytorch` or `tensorrt`"
        )

    # 3.  Generate config.pbtxt
    cfg_text = make_config(
        name=triton_model_name,
        max_batch_size=max_batch_size,
    )
    cfg_file.write_text(cfg_text)

    model_volume.commit()  # persist for future containers
    print(f"Wrote {artifact.name} + config for backend='{TRITON_BACKEND}'")


# `make_config` creates the `config.pbtxt` config file that is necessary
# for Triton to run. Be sure to see their documentation if you
# are optimizing Triton for a specific model.
def make_config(
    name: str,
    max_batch_size: int = 1,
    dtype: str = "TYPE_FP16",
) -> str:
    """
    Construct a minimal `config.pbtxt` string.

    Important fields:
    - name            : model name (dir name) inside the repo.
    - backend         : "pytorch" or "tensorrt".
    - max_batch_size  : Triton scheduler batch limit (runtime can form micro-batches up to this).
    - input/output    : names must match the saved artifact (TorchScript/ONNX/TRT).
    - instance_group  : number of model instances per GPU (intra-GPU concurrency).

    Return
    ------
    str : `config.pbtxt` contents to be written to disk.
    """
    from textwrap import dedent

    # Config basics: choose a backend
    cfg = f"""\
        name: "{name}"
        backend: "{TRITON_BACKEND}"
        max_batch_size: {max_batch_size}
        """
    # Set inputs/outputs info
    cfg += f"""\
        input [
        {{
            name: "{IN_NAME}"
            data_type: {dtype}
            dims: [ {", ".join(map(str, INPUT_SHAPE))} ]
        }}
        ]

        output [
        {{
            name: "{OUT_NAME}"
            data_type: {dtype}
            dims: [ {OUTPUT_DIM} ]
        }}
        ]
        """
    # Multi-model concurrency within a single (each) GPU
    cfg += f"""
        instance_group [
        {{ kind: KIND_GPU, count: {N_INSTANCES} }}
        ]
        """

    cfg += f"""
        optimization {{ execution_accelerators {{
        gpu_execution_accelerator : [ {{
            name : "{TRITON_BACKEND}"
            parameters {{ key: "precision_mode" value: "{dtype}" }}
            parameters {{ key: "max_workspace_size_bytes" value: "1073741824" }}
            }}]
        }}}}
        """
    return dedent(cfg)


# # Triton Server app
# Now we actually implement the class which wraps the server.
# The `_start_triton` method is decorated with `@modal.enter()`,
# so it will be run once per container at startup. This method
# launches a Triton instance and waits on a heartbeat check.
# The `preprocessing` and `triton_inference` methods prepare an image
# and pass it to the server. The `embed_bytes` and `embed_url` FastAPI
# endpoints are two entrypoints that allow you to pass an image in
# different ways to the server.
#
# To try it, fire up this Triton server with
# ```python
# modal deploy triton_server.py
# ```
# Then send an image to it with a curl command:
# ```python
# curl -sS -G "$YOUR_EMBED_URL_ENDPOINT" --data-urlencode "url=https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
# ```


@app.cls(
    image=TRITON_IMAGE,
    volumes={VOL_MNT: model_volume},
    cpu=4,
    memory=2.5 * 1024,  # MB -> GB
    gpu="H100:1",
    max_containers=1,
)
@modal.concurrent(max_inputs=2)
class TritonServer:
    @modal.enter()
    def _start_triton(self):
        """
        Build (if needed) the model repo on the mounted Volume, start `tritonserver`,
        then wait for the model to report READY over gRPC.
        """
        self.triton_model_name = TRITON_BACKEND + "::" + MODEL_NAME.replace("/", "_")

        # TODO: sumn w repo_path
        build_triton_repo(
            triton_model_name=self.triton_model_name,
            version="1",
        )

        # Launch Triton. Default ports: HTTP=8000, gRPC=8001, metrics=8002.
        self._proc = subprocess.Popen(
            [
                "tritonserver",
                f"--model-repository={MODEL_REPO}",
                "--exit-on-error=true",
                "--model-control-mode=none",
            ]
        )

        # Heartbeat
        self._client = grpcclient.InferenceServerClient(url="localhost:8001")
        self.wait_for_server()

    def wait_for_server(self, minutes_wait: float = 5):
        """
        Poll gRPC _client for model readiness.
        """
        check_rate_hz = 2
        n_iter = minutes_wait * 60 * check_rate_hz
        for idx in tqdm(
            range(n_iter), total=n_iter, desc="[Modal App]: Waiting for server hearbeat"
        ):
            try:
                if self._client.is_model_ready(self.triton_model_name):
                    break
            except Exception:
                pass
            time.sleep(1 / check_rate_hz)
            if (idx / check_rate_hz) == int(idx / check_rate_hz):
                print(".", end="")
        else:
            raise RuntimeError("Triton failed to become ready")

        print("Triton ready for inference!")

    def preprocess(self, image_bytes: bytes):
        """
        Convert image bytes into Numpy array for Triton consumption.
        """
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        u8 = np.asarray(img, dtype=np.uint8)  # [H, W, 3], uint8

        # Allocate final tensor [N, C, H, W] in float16
        out = np.empty((1, 3, 224, 224), dtype=np.float16)

        # Cast + scale + transpose in a single pass into 'out'
        np.multiply(
            u8.transpose(2, 0, 1)[None, ...],  # view: [1, 3, H, W]
            np.float16(1.0 / 255.0),
            out=out,
            casting="unsafe",  # allow u8 -> f16
        )
        return out

    def triton_inference(self, image_bytes: bytes) -> dict:
        t0 = time.perf_counter()
        input_arr = self.preprocess(image_bytes)
        t_prep = time.perf_counter() - t0

        # Build request
        inp = grpcclient.InferInput(IN_NAME, input_arr.shape, "FP16")
        inp.set_data_from_numpy(input_arr)

        out = grpcclient.InferRequestedOutput(OUT_NAME)

        # Inference
        t1 = time.perf_counter()
        resp = self._client.infer(self.triton_model_name, [inp], outputs=[out])
        t_inf = time.perf_counter() - t1

        return {
            "preprocessing-time": t_prep,
            "inference-time": t_inf,
            "embedding": resp.as_numpy(OUT_NAME)[0].tolist(),
        }

    @modal.fastapi_endpoint(
        method="POST",
        label="embed-bytes",
        docs=True,
    )
    async def embed_bytes(
        self,
        image_bytes: bytes = FastAPIBody(..., media_type="application/octet-stream"),
    ):
        self.triton_inference(image_bytes)

    @modal.fastapi_endpoint(
        method="GET",
        label="embed-url",
        docs=True,
    )
    async def embed_url(self, url: str) -> dict:
        img = Image.open(urlopen(url))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        return self.triton_inference(image_bytes)

    @modal.exit()
    def _cleanup(self):
        if hasattr(self, "_proc"):
            self._proc.terminate()
