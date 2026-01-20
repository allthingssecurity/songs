"""
RunPod Serverless Handler for HeartMuLa Music Generation

Input format:
{
    "input": {
        "lyrics": "string - the lyrics text or path concept",
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
import torch
import runpod

# Add parent directory to path for heartlib imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heartlib import HeartMuLaGenPipeline

# Global model instance for warm starts
MODEL = None
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/ckpt")


def load_model():
    """Load the HeartMuLa model (called once on cold start)."""
    global MODEL

    if MODEL is not None:
        return MODEL

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

        # Calculate approximate duration
        # 48kHz sample rate, mp3 at ~128kbps
        duration_ms = max_audio_length_ms  # Approximate

        return {
            "audio_base64": audio_base64,
            "duration_ms": duration_ms,
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
    # Test with sample input
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
        print(f"Duration: {result['duration_ms']}ms, Sample rate: {result['sample_rate']}Hz")

        # Optionally save the output for testing
        with open("test_output.mp3", "wb") as f:
            f.write(base64.b64decode(result["audio_base64"]))
        print("Saved to test_output.mp3")


# RunPod serverless entry point
runpod.serverless.start({"handler": handler})
