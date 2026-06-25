import os
import requests
import logging

logger = logging.getLogger("download_models")
logging.basicConfig(level=logging.INFO)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

GIPFORMER_FILES = {
    "encoder.onnx": "https://huggingface.co/g-group-ai-lab/gipformer-65M-rnnt/resolve/main/encoder-epoch-35-avg-6.onnx",
    "decoder.onnx": "https://huggingface.co/g-group-ai-lab/gipformer-65M-rnnt/resolve/main/decoder-epoch-35-avg-6.onnx",
    "joiner.onnx": "https://huggingface.co/g-group-ai-lab/gipformer-65M-rnnt/resolve/main/joiner-epoch-35-avg-6.onnx",
    "tokens.txt": "https://huggingface.co/g-group-ai-lab/gipformer-65M-rnnt/resolve/main/tokens.txt",
}

VAD_FILES = {
    "silero_vad.onnx": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
}

def download_file(url: str, dest_path: str):
    logger.info(f"Downloading {url} -> {dest_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    downloaded = 0
    
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0 and downloaded % (1024 * 1024 * 10) == 0: # Log progress every 10MB
                    percent = (downloaded / total_size) * 100
                    logger.info(f"Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")

def download_all_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Download Gipformer files
    gipformer_dir = os.path.join(MODELS_DIR, "gipformer")
    os.makedirs(gipformer_dir, exist_ok=True)
    for filename, url in GIPFORMER_FILES.items():
        dest = os.path.join(gipformer_dir, filename)
        if not os.path.exists(dest):
            logger.info(f"Model file {filename} missing. Starting download...")
            download_file(url, dest)
        else:
            logger.info(f"Model file {filename} already exists at {dest}")

    # Download Silero VAD
    vad_dir = os.path.join(MODELS_DIR, "vad")
    os.makedirs(vad_dir, exist_ok=True)
    for filename, url in VAD_FILES.items():
        dest = os.path.join(vad_dir, filename)
        if not os.path.exists(dest):
            logger.info(f"VAD file {filename} missing. Starting download...")
            download_file(url, dest)
        else:
            logger.info(f"VAD file {filename} already exists at {dest}")

if __name__ == "__main__":
    download_all_models()
