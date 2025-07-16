from lib import hello_world
import modal

app = modal.App("auto-mount")

@app.function()
def my_func():
    print(hello_world())
