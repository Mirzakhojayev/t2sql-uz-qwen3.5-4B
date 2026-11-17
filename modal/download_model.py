"""Run once: copies Qwen3.5-4B (public, about 9 GB) from Hugging Face into this Modal account's volume. CPU only, a few cents.

  modal run modal/download_model.py
"""
import os
import time

import modal

app = modal.App("t2sql-download")
vol = modal.Volume.from_name("t2sql-cache", create_if_missing=True)
MODEL_DIR = "/cache/models/Qwen3.5-4B"


@app.function(image=modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub"),
              volumes={"/cache": vol}, timeout=1800, cpu=4)
def download() -> str:
    from huggingface_hub import snapshot_download

    t = time.time()
    snapshot_download("Qwen/Qwen3.5-4B", local_dir=MODEL_DIR)
    vol.commit()
    size = sum(os.path.getsize(os.path.join(MODEL_DIR, f)) for f in os.listdir(MODEL_DIR) if not f.startswith("."))
    return f"downloaded in {time.time() - t:.0f}s, {size / 1e9:.1f} GB"


@app.local_entrypoint()
def main():
    print(download.remote())
