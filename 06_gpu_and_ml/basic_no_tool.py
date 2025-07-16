from typing import Any

import modal

cuda_version = "12.4.0" 
flavor = "devel" 
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

hf_cache_dir = "/root/.cache/huggingface"
vllm_cache_dir = "/root/.cache/vllm"
SERVER_DEFAULT = "http://0.0.0.0:8000"

vllm_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .pip_install(
        "uv",
        "hf_transfer"
    ).run_commands(
        "uv pip install --system -U 'vllm[audio]' --extra-index-url https://wheels.vllm.ai/nightly"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": hf_cache_dir, "VLLM_CACHE_DIR": vllm_cache_dir, "VLLM_TORCH_COMPILE_LEVEL": "O1"}) 
)

app = modal.App(name="send-message")
HF_CACHE_VOLUME = modal.Volume.from_name("send-message-cache", create_if_missing=True)
VLLM_CACHE_VOLUME = modal.Volume.from_name("send-message-vllm-cache", create_if_missing=True)

@app.function(gpu="H100", image=vllm_image, timeout=60 * 10, volumes={hf_cache_dir: HF_CACHE_VOLUME, vllm_cache_dir: VLLM_CACHE_VOLUME})
def send_message():
    import subprocess
    import requests
    import time

    process = subprocess.Popen([
        "vllm", 
        "serve", 
        "mistralai/Voxtral-Mini-3B-2507",
        "--tokenizer_mode", "mistral",
        "--config_format", "mistral", 
        "--load_format", "mistral",
        "--tool-call-parser", "mistral",
        "--enable-auto-tool-choice"
    ])

    poll = True
    while poll:
        try:
            response = requests.get(f"{SERVER_DEFAULT}/health")
            if response.status_code == 200:
                poll = False
        except requests.exceptions.ConnectionError:
            print("Server not ready yet, waiting for 10 second before retrying...")
            time.sleep(10)
    

    from mistral_common.protocol.instruct.messages import TextChunk, AudioChunk, UserMessage, AssistantMessage, RawAudio
    from mistral_common.audio import Audio
    from huggingface_hub import hf_hub_download

    from openai import OpenAI

    openai_api_key = "EMPTY"
    openai_api_base = f"{SERVER_DEFAULT}/v1"

    client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    obama_file = hf_hub_download("patrickvonplaten/audio_samples", "obama.mp3", repo_type="dataset")
    bcn_file = hf_hub_download("patrickvonplaten/audio_samples", "bcn_weather.mp3", repo_type="dataset")

    def file_to_chunk(file: str) -> AudioChunk:
        audio = Audio.from_file(file, strict=False)
        return AudioChunk.from_audio(audio)

    text_chunk = TextChunk(text="Which speaker is more inspiring? Why? How are they different from each other?")
    user_msg = UserMessage(content=[file_to_chunk(obama_file), file_to_chunk(bcn_file), text_chunk]).to_openai()

    print(30 * "=" + "USER 1" + 30 * "=")
    print(text_chunk.text)
    print("\n\n")

    response = client.chat.completions.create(
        model=model,
        messages=[user_msg],
        temperature=0.2,
        top_p=0.95,
    )
    content = response.choices[0].message.content

    print(30 * "=" + "BOT 1" + 30 * "=")
    print(content)
    print("\n\n")

    messages = [
        user_msg,
        AssistantMessage(content=content).to_openai(),
        UserMessage(content="Ok, now please summarize the content of the first audio.").to_openai()
    ]
    print(30 * "=" + "USER 2" + 30 * "=")
    print(messages[-1]["content"])
    print("\n\n")

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        top_p=0.95,
    )
    content = response.choices[0].message.content
    print(30 * "=" + "BOT 2" + 30 * "=")
    print(content)


@app.local_entrypoint()
def main():
    print(send_message.remote())