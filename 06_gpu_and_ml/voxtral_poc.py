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
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": hf_cache_dir, "VLLM_CACHE_DIR": vllm_cache_dir, "VLLM_TORCH_COMPILE_LEVEL": "0"}) 
)

app = modal.App(name="send-message")
HF_CACHE_VOLUME = modal.Volume.from_name("send-message-cache", create_if_missing=True)
VLLM_CACHE_VOLUME = modal.Volume.from_name("send-message-vllm-cache", create_if_missing=True)

@app.function(gpu="H100", image=vllm_image, timeout=60 * 20, volumes={hf_cache_dir: HF_CACHE_VOLUME, vllm_cache_dir: VLLM_CACHE_VOLUME})
def send_message():
    import subprocess
    import requests
    import time
    import json

    process = subprocess.Popen([
        "vllm", 
        "serve", 
        "mistralai/Voxtral-Small-24B-2507",
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
            print("Server not ready yet, waiting for 20 seconds before retrying...")
            time.sleep(20)
    
    from mistral_common.protocol.instruct.messages import TextChunk, AudioChunk, UserMessage, AssistantMessage
    from mistral_common.audio import Audio
    from huggingface_hub import hf_hub_download
    from mistral_common.protocol.instruct.tool_calls import Tool, Function
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

    def preprocess_text(text: str) -> str:
        """Convert text to lowercase"""
        return text.lower()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "preprocess_text",
                "description": "Convert text to lowercase for preprocessing",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to convert to lowercase",
                        }
                    },
                    "required": ["text"],
                },
            },
        }
    ]

    names_to_functions = {
        'preprocess_text': preprocess_text
    }

    print(30 * "=" + "Function calling" + 30 * "=")

    tool_messages = [
        {"role": "user", "content": f"Please preprocess this text: '{content}'"}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=tool_messages,
        temperature=0.2,
        top_p=0.95,
        tools=tools,
        tool_choice="required"
    )

    print(30 * "=" + "BOT 3" + 30 * "=")
    print(response.choices[0].message.content)
    print(response.choices[0].message.tool_calls)

    if response.choices[0].message.tool_calls:
        tool_messages.append({
            "role": "assistant",
            "content": response.choices[0].message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in response.choices[0].message.tool_calls
            ]
        })

        for tool_call in response.choices[0].message.tool_calls:
            function_name = tool_call.function.name
            function_params = json.loads(tool_call.function.arguments)
            
            print(f"Executing function: {function_name}")
            print(f"With parameters: {function_params}")
            
            function_result = names_to_functions[function_name](**function_params)
            print(f"Function result: {function_result}")
            
            tool_messages.append({
                "role": "tool",
                "name": function_name,
                "content": function_result,
                "tool_call_id": tool_call.id
            })

        final_response = client.chat.completions.create(
            model=model,
            messages=tool_messages,
            temperature=0.2,
            top_p=0.95,
        )
        
        print(30 * "=" + "FINAL RESPONSE" + 30 * "=")
        print(final_response.choices[0].message.content)
    else:
        print("No tool was called")

@app.local_entrypoint()
def main():
    print(send_message.remote())