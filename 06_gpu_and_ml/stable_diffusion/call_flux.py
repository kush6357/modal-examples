import time
from pathlib import Path

import modal

if __name__ == "__main__":
    APP_NAME = "example-flux"
    Model = modal.Cls.from_name(app_name=APP_NAME, name="Model")

    prompt: str = "a computer screen showing ASCII terminal art of the"
    " word 'Modal' in neon green. two programmers are pointing excitedly"
    (" at the screen.",)
    twice: bool = (True,)
    compile: bool = False

    t0 = time.time()
    image_bytes = Model(compile=compile).inference.remote(prompt)
    print(f"🎨 first inference latency: {time.time() - t0:.2f} seconds")

    if twice:
        t0 = time.time()
        image_bytes = Model(compile=compile).inference.remote(prompt)
        print(f"🎨 second inference latency: {time.time() - t0:.2f} seconds")

    output_path = Path("/tmp") / "flux" / "output.jpg"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    print(f"🎨 saving output to {output_path}")
    output_path.write_bytes(image_bytes)
