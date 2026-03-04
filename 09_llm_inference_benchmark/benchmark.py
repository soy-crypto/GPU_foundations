import subprocess
import yaml

MODELS_CONFIG = "configs/models.yaml"

with open(MODELS_CONFIG) as f:
    models = yaml.safe_load(f)["models"]

backends = [
    "hf",
    "vllm",
    "trtllm"
]

for backend in backends:
    for model in models:

        print(f"Running {backend} on {model}")

        subprocess.run([
            "python",
            f"run_{backend}.py",
            model
        ])