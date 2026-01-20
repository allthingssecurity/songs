"""
RunPod Serverless Handler for HeartMuLa Music Generation

Downloads model weights on first cold start, then caches them.

Input format:
{
    "input": {
        "lyrics": "string - the lyrics text",
        "tags": "string - comma-separated tags (optional, default: 'pop,upbeat')",
        "max_audio_length_ms": int - max audio length in ms (optional, default: 120000),
        "temperature": float - sampling temperature (optional, default: 1.0),
        "topk": int - top-k sampling (optional, default: 50),
        "cfg_scale": float - classifier-free guidance scale (optional, default: 1.5)
    }
}

Output format:
{
    "audio_base64": "base64 encoded mp3 audio",
    "duration_ms": int,
    "sample_rate": int,
    "format": "mp3"
}
"""

import os
import sys
import base64
import tempfile
import subprocess

# Add parent directory to path for heartlib imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/src")

# Global model instance for warm starts
MODEL = None
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/ckpt")


def download_models():
    """Download model weights from HuggingFace if not present."""

    heartmula_path = os.path.join(MODEL_PATH, "HeartMuLa-oss-3B")
    heartcodec_path = os.path.join(MODEL_PATH, "HeartCodec-oss")
    tokenizer_path = os.path.join(MODEL_PATH, "tokenizer.json")

    # Check if models already exist
    if (os.path.exists(heartmula_path) and
        os.path.exists(heartcodec_path) and
        os.path.exists(tokenizer_path)):
        print("Models already downloaded.")
        return

    print("Downloading model weights from HuggingFace...")

    # Download HeartMuLaGen (tokenizer and config)
    print("Downloading HeartMuLaGen base files...")
    subprocess.run([
        "huggingface-cli", "download",
        "HeartMuLa/HeartMuLaGen",
        "--local-dir", MODEL_PATH
    ], check=True)

    # Download HeartMuLa-oss-3B
    print("Downloading HeartMuLa-oss-3B (this may take a while)...")
    subprocess.run([
        "huggingface-cli", "download",
        "HeartMuLa/HeartMuLa-oss-3B",
        "--local-dir", heartmula_path
    ], check=True)

    # Download HeartCodec-oss
    print("Downloading HeartCodec-oss...")
    subprocess.run([
        "huggingface-cli", "download",
        "HeartMuLa/HeartCodec-oss",
        "--local-dir", heartcodec_path
    ], check=True)

    print("All models downloaded successfully!")


def load_model():
    """Load the HeartMuLa model (called once on cold start)."""
    global MODEL

    if MODEL is not None:
        return MODEL

    # Download models if needed
    download_models()

    import torch
    from heartlib import HeartMuLaGenPipeline

    print("Loading HeartMuLa model...")

    # Use CUDA with bfloat16 for optimal performance
    device = torch.device("cuda")
    dtype = torch.bfloat16

    MODEL = HeartMuLaGenPipeline.from_pretrained(
        MODEL_PATH,
        device=device,
        dtype=dtype,
        version="3B",
    )

    print("Model loaded successfully!")
    return MODEL


def generate_music(lyrics: str, tags: str = "pop,upbeat", **kwargs) -> dict:
    """
    Generate music from lyrics and tags.

    Args:
        lyrics: The lyrics text
        tags: Comma-separated tags for style/genre
        **kwargs: Additional generation parameters

    Returns:
        dict with audio_base64, duration_ms, sample_rate, format
    """
    import torch

    pipe = load_model()

    # Default parameters
    max_audio_length_ms = kwargs.get("max_audio_length_ms", 120000)
    temperature = kwargs.get("temperature", 1.0)
    topk = kwargs.get("topk", 50)
    cfg_scale = kwargs.get("cfg_scale", 1.5)

    # Create temporary file for output
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
        output_path = tmp_file.name

    try:
        # Generate music
        with torch.no_grad():
            pipe(
                {
                    "lyrics": lyrics,
                    "tags": tags,
                },
                max_audio_length_ms=max_audio_length_ms,
                save_path=output_path,
                topk=topk,
                temperature=temperature,
                cfg_scale=cfg_scale,
            )

        # Read and encode the output
        with open(output_path, "rb") as f:
            audio_data = f.read()

        audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        return {
            "audio_base64": audio_base64,
            "duration_ms": max_audio_length_ms,
            "sample_rate": 48000,
            "format": "mp3",
            "size_bytes": len(audio_data),
        }

    finally:
        # Cleanup temp file
        if os.path.exists(output_path):
            os.remove(output_path)


def handler(job):
    """
    RunPod serverless handler function.

    Args:
        job: RunPod job object containing input data

    Returns:
        dict with generation results or error
    """
    try:
        job_input = job.get("input", {})

        # Validate required input
        lyrics = job_input.get("lyrics")
        if not lyrics:
            return {"error": "Missing required field: 'lyrics'"}

        # Optional parameters
        tags = job_input.get("tags", "pop,upbeat")
        max_audio_length_ms = job_input.get("max_audio_length_ms", 120000)
        temperature = job_input.get("temperature", 1.0)
        topk = job_input.get("topk", 50)
        cfg_scale = job_input.get("cfg_scale", 1.5)

        # Validate parameters
        if max_audio_length_ms > 240000:
            return {"error": "max_audio_length_ms cannot exceed 240000 (4 minutes)"}
        if max_audio_length_ms < 10000:
            return {"error": "max_audio_length_ms must be at least 10000 (10 seconds)"}

        # Generate music
        result = generate_music(
            lyrics=lyrics,
            tags=tags,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk,
            cfg_scale=cfg_scale,
        )

        return result

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# For local testing
if __name__ == "__main__":
    import runpod

    # Test locally if no runpod
    if os.environ.get("RUNPOD_POD_ID"):
        runpod.serverless.start({"handler": handler})
    else:
        # Local test
        test_job = {
            "input": {
                "lyrics": """[Verse]
Hello world, this is a test
Of the music generation
Creating songs with AI
Making melodies fly high

[Chorus]
Music in the cloud
Singing out loud
RunPod generation
A new creation""",
                "tags": "pop,electronic,upbeat",
                "max_audio_length_ms": 60000,
            }
        }

        result = handler(test_job)

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Success! Generated {result['size_bytes']} bytes of audio")

            # Save the output
            with open("test_output.mp3", "wb") as f:
                f.write(base64.b64decode(result["audio_base64"]))
            print("Saved to test_output.mp3")
