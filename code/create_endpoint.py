import modal
import os
import time
import subprocess
import threading

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install(
        "vllm==0.7.3",
        "huggingface-hub==0.26.0",
        "hf_transfer",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_USE_V1": "0",
        }
    )
)

app = modal.App("llama-reviewer-functional")
volume = modal.Volume.from_name("reviewer-storage")


@app.function(volumes={"/data": volume})
def fix_tokenizer_config():
    import json

    config_path = "/data/awq-model/tokenizer_config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        if config.get("tokenizer_class") == "TokenizersBackend":
            print("Fixing TokenizersBackend typo...")
            config["tokenizer_class"] = "PreTrainedTokenizerFast"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            volume.commit()  # CRITICAL: Saves changes to the cloud volume
            print("Fix applied and committed.")
        else:
            print(f"No fix needed. Current class: {config.get('tokenizer_class')}")
    else:
        print(f"Path not found: {config_path}")


@app.function(
    image=vllm_image,
    max_containers=1,
    volumes={"/data": volume},
    timeout=3600,
)
def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="maxidl/Llama-OpenReviewer-8B",
        local_dir="/data/Llama-OpenReviewer-8B",
    )
    volume.commit()


@app.cls(
    image=vllm_image,
    gpu="A100",
    max_containers=1,
    volumes={"/data": volume},
    scaledown_window=300,
    timeout=1800,
)
class VLLMServer:
    @modal.enter()
    def start_vllm(self):
        cmd = [
            "python3",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            "/data/Llama-OpenReviewer-8B",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "12288",
            "--trust-remote-code",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--gpu-memory-utilization",
            "0.95",
        ]

        print("Starting vLLM Engine...", flush=True)
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )

        self.ready = False

        def log_dumper():
            for line in self.process.stdout:
                print(f"(vLLM) {line.strip()}", flush=True)
                if "Application startup complete" in line:
                    self.ready = True

        threading.Thread(target=log_dumper, daemon=True).start()

        start_time = time.time()
        while not self.ready:
            if time.time() - start_time > 600:
                print("vLLM failed to start within timeout.")
                break
            time.sleep(1)

        print("vLLM is ready to handle requests!")

    @modal.web_server(8000)
    def vllm_endpoint(self):
        pass
