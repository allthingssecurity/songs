"""
RunPod Client for HeartMuLa Music Generation

Usage:
    python client.py --lyrics "Your lyrics here" --tags "pop,rock" --output output.mp3

Environment Variables:
    RUNPOD_API_KEY: Your RunPod API key
    RUNPOD_ENDPOINT_ID: Your deployed endpoint ID
"""

import os
import sys
import time
import base64
import argparse
import requests
from typing import Optional


class HeartMuLaClient:
    """Client for HeartMuLa RunPod serverless endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint_id: Optional[str] = None,
    ):
        """
        Initialize the client.

        Args:
            api_key: RunPod API key (or set RUNPOD_API_KEY env var)
            endpoint_id: RunPod endpoint ID (or set RUNPOD_ENDPOINT_ID env var)
        """
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        self.endpoint_id = endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID")

        if not self.api_key:
            raise ValueError("RunPod API key required. Set RUNPOD_API_KEY or pass api_key.")
        if not self.endpoint_id:
            raise ValueError("RunPod endpoint ID required. Set RUNPOD_ENDPOINT_ID or pass endpoint_id.")

        self.base_url = f"https://api.runpod.ai/v2/{self.endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate(
        self,
        lyrics: str,
        tags: str = "pop,upbeat",
        max_audio_length_ms: int = 120000,
        temperature: float = 1.0,
        topk: int = 50,
        cfg_scale: float = 1.5,
        timeout: int = 300,
        poll_interval: int = 5,
    ) -> dict:
        """
        Generate music from lyrics.

        Args:
            lyrics: The lyrics text
            tags: Comma-separated style/genre tags
            max_audio_length_ms: Maximum audio length in milliseconds
            temperature: Sampling temperature
            topk: Top-k sampling parameter
            cfg_scale: Classifier-free guidance scale
            timeout: Maximum time to wait for completion (seconds)
            poll_interval: Time between status checks (seconds)

        Returns:
            dict with audio_base64, duration_ms, sample_rate, format
        """
        # Submit the job
        payload = {
            "input": {
                "lyrics": lyrics,
                "tags": tags,
                "max_audio_length_ms": max_audio_length_ms,
                "temperature": temperature,
                "topk": topk,
                "cfg_scale": cfg_scale,
            }
        }

        response = requests.post(
            f"{self.base_url}/run",
            headers=self.headers,
            json=payload,
        )
        response.raise_for_status()
        result = response.json()

        job_id = result.get("id")
        if not job_id:
            raise RuntimeError(f"No job ID returned: {result}")

        print(f"Job submitted: {job_id}")

        # Poll for completion
        start_time = time.time()
        while time.time() - start_time < timeout:
            status_response = requests.get(
                f"{self.base_url}/status/{job_id}",
                headers=self.headers,
            )
            status_response.raise_for_status()
            status = status_response.json()

            job_status = status.get("status")
            print(f"Status: {job_status}")

            if job_status == "COMPLETED":
                output = status.get("output", {})
                if "error" in output:
                    raise RuntimeError(f"Generation error: {output['error']}")
                return output

            elif job_status == "FAILED":
                error = status.get("error", "Unknown error")
                raise RuntimeError(f"Job failed: {error}")

            elif job_status in ["IN_QUEUE", "IN_PROGRESS"]:
                time.sleep(poll_interval)

            else:
                raise RuntimeError(f"Unknown status: {job_status}")

        raise TimeoutError(f"Job {job_id} timed out after {timeout} seconds")

    def generate_sync(
        self,
        lyrics: str,
        tags: str = "pop,upbeat",
        max_audio_length_ms: int = 120000,
        temperature: float = 1.0,
        topk: int = 50,
        cfg_scale: float = 1.5,
        timeout: int = 300,
    ) -> dict:
        """
        Generate music synchronously (uses /runsync endpoint).

        Note: This blocks until completion, best for shorter generations.
        """
        payload = {
            "input": {
                "lyrics": lyrics,
                "tags": tags,
                "max_audio_length_ms": max_audio_length_ms,
                "temperature": temperature,
                "topk": topk,
                "cfg_scale": cfg_scale,
            }
        }

        response = requests.post(
            f"{self.base_url}/runsync",
            headers=self.headers,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()

        if result.get("status") == "COMPLETED":
            output = result.get("output", {})
            if "error" in output:
                raise RuntimeError(f"Generation error: {output['error']}")
            return output
        else:
            raise RuntimeError(f"Unexpected result: {result}")

    def save_audio(self, result: dict, output_path: str):
        """Save the generated audio to a file."""
        audio_base64 = result.get("audio_base64")
        if not audio_base64:
            raise ValueError("No audio data in result")

        audio_data = base64.b64decode(audio_base64)
        with open(output_path, "wb") as f:
            f.write(audio_data)

        print(f"Saved audio to: {output_path}")
        print(f"Size: {len(audio_data)} bytes")
        print(f"Format: {result.get('format', 'unknown')}")
        print(f"Sample rate: {result.get('sample_rate', 'unknown')} Hz")


def main():
    parser = argparse.ArgumentParser(description="Generate music with HeartMuLa on RunPod")
    parser.add_argument("--lyrics", "-l", required=True, help="Lyrics text or path to lyrics file")
    parser.add_argument("--tags", "-t", default="pop,upbeat", help="Comma-separated tags")
    parser.add_argument("--output", "-o", default="output.mp3", help="Output file path")
    parser.add_argument("--max-length", type=int, default=120000, help="Max audio length in ms")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--topk", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--cfg-scale", type=float, default=1.5, help="CFG scale")
    parser.add_argument("--api-key", help="RunPod API key (or set RUNPOD_API_KEY)")
    parser.add_argument("--endpoint-id", help="RunPod endpoint ID (or set RUNPOD_ENDPOINT_ID)")
    parser.add_argument("--sync", action="store_true", help="Use synchronous endpoint")

    args = parser.parse_args()

    # Read lyrics from file if path provided
    lyrics = args.lyrics
    if os.path.isfile(lyrics):
        with open(lyrics, "r") as f:
            lyrics = f.read()

    # Create client
    client = HeartMuLaClient(
        api_key=args.api_key,
        endpoint_id=args.endpoint_id,
    )

    # Generate music
    print("Generating music...")
    if args.sync:
        result = client.generate_sync(
            lyrics=lyrics,
            tags=args.tags,
            max_audio_length_ms=args.max_length,
            temperature=args.temperature,
            topk=args.topk,
            cfg_scale=args.cfg_scale,
        )
    else:
        result = client.generate(
            lyrics=lyrics,
            tags=args.tags,
            max_audio_length_ms=args.max_length,
            temperature=args.temperature,
            topk=args.topk,
            cfg_scale=args.cfg_scale,
        )

    # Save audio
    client.save_audio(result, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
