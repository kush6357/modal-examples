import os
import subprocess
import modal
import modal.experimental
import secrets

image = (
    modal.Image.debian_slim()
    .pip_install("ray[default]", "jupyterlab", "ipywidgets")
    .add_local_file("ray_start.ipynb", remote_path="/root/ray_start.ipynb")
)
app = modal.App(image=image)


@app.function(cpu=4, memory=1024 * 8, timeout=3600)
@modal.experimental.clustered(2)
def hello():
    cluster_info = modal.experimental.get_cluster_info()
    rank = cluster_info.rank
    if rank == 0:
        with modal.forward(8265) as tunnel, modal.forward(8888) as jupyter_tunnel:
            subprocess.run(
                "ray start --head --disable-usage-stats --port=6379 --num-cpus=4 --dashboard-host=0.0.0.0",
                shell=True,
            )

            token = secrets.token_urlsafe(13)
            url = jupyter_tunnel.url + "/?token=" + token + "lab/workspaces/auto-q/tree/ray_start.ipynb"
            print(f"Ray dashboard: {tunnel.url}")
            print(f"Starting Jupyter at {url}")
            subprocess.run(
                [
                    "jupyter",
                    "lab",
                    "--no-browser",
                    "--allow-root",
                    "--ip=0.0.0.0",
                    "--port=8888",
                    "--LabApp.allow_origin='*'",
                    "--LabApp.allow_remote_access=1",
                ],
                env={**os.environ, "JUPYTER_TOKEN": token, "SHELL": "/bin/bash"},
                stderr=subprocess.DEVNULL,
            )
    else:
        subprocess.run("ray start --disable-usage-stats --address=10.100.0.1:6379 --block --num-cpus=4", shell=True)
