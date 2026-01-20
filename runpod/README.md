# HeartMuLa RunPod Serverless Deployment

Self-contained Docker image with model weights included. Just build, push, and deploy.

## Quick Deploy (3 Steps)

### 1. Build Docker Image

```bash
# From heartlib root directory
cd /path/to/heartlib

# Build (takes ~15-20 min, downloads ~15GB of model weights)
docker build -t YOUR_DOCKERHUB_USER/heartmula:latest -f runpod/Dockerfile .

# Push to Docker Hub
docker login
docker push YOUR_DOCKERHUB_USER/heartmula:latest
```

### 2. Create RunPod Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click **"New Endpoint"**
3. Configure:
   - **Name**: `heartmula-music`
   - **Select a Template**: Choose **"Custom"**
   - **Docker Image**: `YOUR_DOCKERHUB_USER/heartmula:latest`
   - **GPU**: Select 24GB+ VRAM (A10, RTX 4090, A100)
   - **Container Disk**: 50GB (for the weights)
   - **Idle Timeout**: 60 seconds
   - **Max Workers**: 3
4. Click **"Create"**
5. Copy your **Endpoint ID**

### 3. Test It

```bash
# Set your credentials
export RUNPOD_API_KEY="your_api_key_here"
export RUNPOD_ENDPOINT_ID="your_endpoint_id_here"

# Generate a song
python runpod/client.py \
  --lyrics "[Verse]\nHello world, testing\n[Chorus]\nMusic generation works" \
  --tags "pop,piano" \
  --output my_song.mp3
```

---

## API Usage

### HTTP Request

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "lyrics": "[Verse]\nSunshine in the morning\nCoffee in my hand\n[Chorus]\nIts a beautiful day\nLets make it grand",
      "tags": "pop,acoustic,happy",
      "max_audio_length_ms": 60000
    }
  }'
```

### Response

```json
{
  "id": "job-abc123",
  "status": "IN_QUEUE"
}
```

### Check Status / Get Result

```bash
curl "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/job-abc123" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Completed Response

```json
{
  "status": "COMPLETED",
  "output": {
    "audio_base64": "<base64 MP3 data>",
    "duration_ms": 60000,
    "sample_rate": 48000,
    "format": "mp3",
    "size_bytes": 960000
  }
}
```

---

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lyrics` | string | **required** | Song lyrics with section markers |
| `tags` | string | `"pop,upbeat"` | Comma-separated style tags |
| `max_audio_length_ms` | int | `120000` | Max audio length (10000-240000) |
| `temperature` | float | `1.0` | Sampling temperature |
| `topk` | int | `50` | Top-k sampling |
| `cfg_scale` | float | `1.5` | Classifier-free guidance |

---

## Lyrics Format

```
[Intro]

[Verse]
First verse lyrics
More lyrics here

[Chorus]
Catchy chorus part
Memorable lines

[Bridge]
Different section

[Outro]
Ending lines
```

---

## Example Tags

- **Genre**: `pop`, `rock`, `electronic`, `jazz`, `hip-hop`, `classical`, `folk`
- **Mood**: `happy`, `sad`, `upbeat`, `melancholic`, `energetic`, `romantic`
- **Instruments**: `piano`, `guitar`, `synthesizer`, `violin`, `drums`

---

## Python Client Usage

```python
from runpod.client import HeartMuLaClient

client = HeartMuLaClient(
    api_key="your_api_key",
    endpoint_id="your_endpoint_id"
)

result = client.generate(
    lyrics="[Verse]\nYour lyrics here",
    tags="pop,piano",
    max_audio_length_ms=60000
)

client.save_audio(result, "output.mp3")
```

---

## Costs

- **Image Size**: ~20GB (includes model weights)
- **GPU Cost**: ~$0.30-0.50/hour
- **Generation**: ~30-60 sec per minute of audio
- **Per Song**: ~$0.01-0.02

---

## Files

| File | Description |
|------|-------------|
| `Dockerfile` | Self-contained image with weights |
| `handler.py` | RunPod serverless handler |
| `client.py` | Python API client |
| `requirements.txt` | Dependencies |

---

## Troubleshooting

**Cold Start Slow?** First request takes 30-60s to load model. Use `idle_timeout: 60` to keep warm.

**Out of Memory?** Use GPU with 24GB+ VRAM or reduce `max_audio_length_ms`.

**Timeout?** Use async `/run` endpoint for songs > 2 minutes.
