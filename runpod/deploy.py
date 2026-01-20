#!/usr/bin/env python3
"""
RunPod Deployment Script for HeartMuLa

This script helps you:
1. Build and push the Docker image to Docker Hub
2. Create a RunPod serverless endpoint
3. Upload model checkpoints to RunPod network volume

Prerequisites:
- Docker installed and logged in to Docker Hub
- RunPod API key
- Model checkpoints downloaded locally

Usage:
    python deploy.py build --docker-user YOUR_DOCKERHUB_USERNAME
    python deploy.py create-endpoint --api-key YOUR_RUNPOD_API_KEY
    python deploy.py upload-model --api-key YOUR_RUNPOD_API_KEY
"""

import os
import sys
import argparse
import subprocess
import requests
import json


def build_and_push_image(docker_user: str, tag: str = "latest"):
    """Build and push Docker image to Docker Hub."""
    image_name = f"{docker_user}/heartmula-runpod:{tag}"

    # Get the root directory (parent of runpod/)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dockerfile_path = os.path.join(root_dir, "runpod", "Dockerfile")

    print(f"Building Docker image: {image_name}")
    print(f"Context: {root_dir}")

    # Build the image
    build_cmd = [
        "docker", "build",
        "-t", image_name,
        "-f", dockerfile_path,
        root_dir
    ]

    result = subprocess.run(build_cmd, check=True)

    print(f"\nPushing image to Docker Hub...")
    push_cmd = ["docker", "push", image_name]
    subprocess.run(push_cmd, check=True)

    print(f"\nImage pushed successfully: {image_name}")
    return image_name


def create_endpoint(api_key: str, image_name: str, endpoint_name: str = "heartmula-music-gen"):
    """Create a RunPod serverless endpoint."""
    url = "https://api.runpod.io/graphql"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # GraphQL mutation to create endpoint
    mutation = """
    mutation createEndpoint($input: EndpointInput!) {
        saveEndpoint(input: $input) {
            id
            name
            templateId
        }
    }
    """

    variables = {
        "input": {
            "name": endpoint_name,
            "templateId": None,  # Using custom Docker image
            "dockerImage": image_name,
            "gpuIds": "AMPERE_24",  # RTX 3090/4090, A10, etc.
            "volumeInGb": 50,  # For model checkpoints
            "containerDiskInGb": 20,
            "volumeMountPath": "/runpod-volume",
            "env": [
                {"key": "MODEL_PATH", "value": "/runpod-volume/ckpt"},
                {"key": "HF_HOME", "value": "/runpod-volume/huggingface"},
            ],
            "idleTimeout": 60,  # Keep warm for 60 seconds
            "scalerType": "QUEUE_DELAY",
            "scalerValue": 1,
            "workersMin": 0,
            "workersMax": 3,
        }
    }

    response = requests.post(
        url,
        headers=headers,
        json={"query": mutation, "variables": variables},
    )
    response.raise_for_status()
    result = response.json()

    if "errors" in result:
        print(f"Error creating endpoint: {result['errors']}")
        return None

    endpoint = result.get("data", {}).get("saveEndpoint", {})
    print(f"Endpoint created successfully!")
    print(f"  ID: {endpoint.get('id')}")
    print(f"  Name: {endpoint.get('name')}")

    return endpoint


def get_instructions():
    """Print deployment instructions."""
    instructions = """
================================================================================
HeartMuLa RunPod Deployment Instructions
================================================================================

1. PREREQUISITES:
   - Docker Desktop installed
   - Docker Hub account
   - RunPod account with API key
   - Model checkpoints downloaded locally (./ckpt directory)

2. BUILD AND PUSH DOCKER IMAGE:

   # Login to Docker Hub
   docker login

   # Build and push (from heartlib root directory)
   cd /path/to/heartlib
   docker build -t YOUR_DOCKERHUB_USER/heartmula-runpod:latest -f runpod/Dockerfile .
   docker push YOUR_DOCKERHUB_USER/heartmula-runpod:latest

3. CREATE RUNPOD NETWORK VOLUME:
   - Go to https://www.runpod.io/console/user/storage
   - Create a new Network Volume (50GB recommended)
   - Note the volume ID

4. UPLOAD MODEL CHECKPOINTS:
   - Create a temporary GPU pod with the network volume attached
   - SSH into the pod
   - Download checkpoints:

     cd /runpod-volume
     mkdir -p ckpt
     huggingface-cli download HeartMuLa/HeartMuLaGen --local-dir ./ckpt
     huggingface-cli download HeartMuLa/HeartMuLa-oss-3B --local-dir ./ckpt/HeartMuLa-oss-3B
     huggingface-cli download HeartMuLa/HeartCodec-oss --local-dir ./ckpt/HeartCodec-oss

   - Terminate the temporary pod (keep the volume)

5. CREATE SERVERLESS ENDPOINT:
   - Go to https://www.runpod.io/console/serverless
   - Click "New Endpoint"
   - Select "Custom" template
   - Enter your Docker image: YOUR_DOCKERHUB_USER/heartmula-runpod:latest
   - Configure:
     * GPU: 24GB+ VRAM recommended (A10, RTX 3090/4090)
     * Container Disk: 20GB
     * Volume: Select your network volume
     * Volume Mount: /runpod-volume
     * Environment Variables:
       - MODEL_PATH=/runpod-volume/ckpt
       - HF_HOME=/runpod-volume/huggingface
     * Idle Timeout: 60 seconds
     * Max Workers: 3 (adjust based on needs)
   - Click "Create"

6. TEST YOUR ENDPOINT:

   export RUNPOD_API_KEY="your_api_key"
   export RUNPOD_ENDPOINT_ID="your_endpoint_id"

   python runpod/client.py --lyrics "Hello world, testing music generation" --tags "pop,electronic" --output test.mp3

7. API USAGE:

   curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \\
     -H "Authorization: Bearer YOUR_API_KEY" \\
     -H "Content-Type: application/json" \\
     -d '{
       "input": {
         "lyrics": "[Verse]\\nHello world, this is a test\\n[Chorus]\\nMusic generation is the best",
         "tags": "pop,upbeat",
         "max_audio_length_ms": 60000
       }
     }'

================================================================================
"""
    print(instructions)


def main():
    parser = argparse.ArgumentParser(description="Deploy HeartMuLa to RunPod")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build and push Docker image")
    build_parser.add_argument("--docker-user", required=True, help="Docker Hub username")
    build_parser.add_argument("--tag", default="latest", help="Image tag")

    # Create endpoint command
    create_parser = subparsers.add_parser("create-endpoint", help="Create RunPod endpoint")
    create_parser.add_argument("--api-key", required=True, help="RunPod API key")
    create_parser.add_argument("--image", required=True, help="Docker image name")
    create_parser.add_argument("--name", default="heartmula-music-gen", help="Endpoint name")

    # Instructions command
    subparsers.add_parser("instructions", help="Show deployment instructions")

    args = parser.parse_args()

    if args.command == "build":
        build_and_push_image(args.docker_user, args.tag)
    elif args.command == "create-endpoint":
        create_endpoint(args.api_key, args.image, args.name)
    elif args.command == "instructions":
        get_instructions()
    else:
        get_instructions()


if __name__ == "__main__":
    main()
