# Deployment Guide

This guide covers deploying NRF models for production use.

## 🚀 Quick Deploy

### Option 1: Docker (Recommended)

```bash
# Build production image
docker build --target production -t nrf:latest .

# Run inference server
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  -v $(pwd)/outputs:/app/outputs \
  nrf:latest
```

### Option 2: Direct Installation

```bash
# Install package
pip install nonlinear-rectified-flows

# Run inference
python -m nrf.inference \
  --checkpoint path/to/model.pt \
  --port 8000
```

## 📦 Model Artifacts

### Checkpoint Structure

```
checkpoints/
├── nrf_spline_best.pt          # Best model checkpoint
├── config.yaml                  # Model configuration
├── vae.pt                       # VAE weights
└── metadata.json                # Training metadata
```

### Loading Models

```python
from nrf import load_model

# Load pretrained model
model = load_model("checkpoints/nrf_spline_best.pt")

# Generate images
images = model.generate(
    prompts=["A red cube on top of a blue sphere"],
    num_steps=4,
    guidance_scale=7.5
)
```

## 🌐 API Server

### FastAPI Server

```python
# server.py
from fastapi import FastAPI
from nrf import load_model

app = FastAPI()
model = load_model("checkpoints/nrf_spline_best.pt")

@app.post("/generate")
async def generate(prompt: str, steps: int = 4):
    images = model.generate([prompt], num_steps=steps)
    return {"images": images}
```

Run server:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

## ☁️ Cloud Deployment

### AWS SageMaker

```python
# deploy_sagemaker.py
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(
    model_data="s3://bucket/model.tar.gz",
    role=role,
    framework_version="2.0",
    py_version="py310",
    entry_point="inference.py"
)

predictor = model.deploy(
    instance_type="ml.g5.xlarge",
    initial_instance_count=1
)
```

### Google Cloud Run

```dockerfile
# Dockerfile.cloudrun
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.2-0

WORKDIR /app
COPY . .
RUN pip install -e .

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 server:app
```

Deploy:

```bash
gcloud run deploy nrf \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure ML

```python
# deploy_azure.py
from azureml.core import Workspace, Model
from azureml.core.webservice import AciWebservice

ws = Workspace.from_config()

model = Model.register(
    workspace=ws,
    model_path="checkpoints/",
    model_name="nrf-spline"
)

aci_config = AciWebservice.deploy_configuration(
    cpu_cores=4,
    memory_gb=16,
    gpu_cores=1
)

service = Model.deploy(
    workspace=ws,
    name="nrf-service",
    models=[model],
    deployment_config=aci_config
)
```

## 🔧 Optimization

### TorchScript Export

```python
# Export to TorchScript for faster inference
model = load_model("checkpoints/nrf_spline_best.pt")
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")
```

### ONNX Export

```python
# Export to ONNX for cross-platform deployment
import torch.onnx

dummy_input = torch.randn(1, 4, 64, 64)
dummy_t = torch.tensor([0.5])
dummy_context = torch.randn(1, 768)

torch.onnx.export(
    model,
    (dummy_input, dummy_t, dummy_context),
    "model.onnx",
    opset_version=14,
    input_names=["x_t", "t", "context"],
    output_names=["velocity"],
    dynamic_axes={
        "x_t": {0: "batch_size"},
        "t": {0: "batch_size"},
        "context": {0: "batch_size"},
        "velocity": {0: "batch_size"}
    }
)
```

### TensorRT Optimization

```python
# Convert to TensorRT for NVIDIA GPUs
import torch_tensorrt

trt_model = torch_tensorrt.compile(
    model,
    inputs=[
        torch_tensorrt.Input((1, 4, 64, 64)),
        torch_tensorrt.Input((1,)),
        torch_tensorrt.Input((1, 768))
    ],
    enabled_precisions={torch.float16}
)
```

## 📊 Monitoring

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram

generation_counter = Counter(
    'nrf_generations_total',
    'Total number of generations'
)

generation_duration = Histogram(
    'nrf_generation_duration_seconds',
    'Generation duration in seconds'
)

@generation_duration.time()
def generate_with_metrics(prompt):
    generation_counter.inc()
    return model.generate([prompt])
```

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nrf.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('nrf')
logger.info(f"Generated image for prompt: {prompt}")
```

## 🔒 Security

### API Authentication

```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/generate")
async def generate(
    prompt: str,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    if not verify_token(credentials.credentials):
        raise HTTPException(status_code=401)
    return model.generate([prompt])
```

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/generate")
@limiter.limit("10/minute")
async def generate(request: Request, prompt: str):
    return model.generate([prompt])
```

## 🎯 Performance Tuning

### Batch Processing

```python
# Process multiple prompts in batches
def batch_generate(prompts, batch_size=8):
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        images = model.generate(batch)
        results.extend(images)
    return results
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def generate_cached(prompt: str, steps: int):
    return model.generate([prompt], num_steps=steps)
```

### GPU Memory Management

```python
import torch

# Clear cache periodically
torch.cuda.empty_cache()

# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Mixed precision inference
with torch.cuda.amp.autocast():
    images = model.generate(prompts)
```

## 📈 Scaling

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nrf-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nrf
  template:
    metadata:
      labels:
        app: nrf
    spec:
      containers:
      - name: nrf
        image: nrf:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        ports:
        - containerPort: 8000
```

### Load Balancing

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: nrf-service
spec:
  type: LoadBalancer
  selector:
    app: nrf
  ports:
  - port: 80
    targetPort: 8000
```

## 🔄 CI/CD Pipeline

### GitHub Actions Deployment

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    tags:
      - 'v*'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build and push Docker image
      run: |
        docker build -t nrf:${{ github.ref_name }} .
        docker push nrf:${{ github.ref_name }}
    
    - name: Deploy to production
      run: |
        kubectl set image deployment/nrf-deployment \
          nrf=nrf:${{ github.ref_name }}
```

## 📞 Support

For deployment issues:
- GitHub Issues: [Report Issue](https://github.com/Todd7777/KURE-Project/issues)
- Email: todd.zhou@example.com
- Discord: [Join Server](https://discord.gg/XXXXXXX)
