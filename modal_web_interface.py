"""
Web interface for KURE Project on Modal
Provides a simple web UI for image generation
"""

import modal
from modal import Image, web_endpoint
import json
import base64
import io
from PIL import Image as PILImage

# Use the same image as the main app
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "wget", 
        "curl",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        "libgcc-s1"
    )
    .pip_install([
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "diffusers>=0.21.0",
        "accelerate>=0.24.0",
        "transformers>=4.35.0",
        "open_clip_torch>=2.20.0",
        "pillow>=10.0.0",
        "fastapi>=0.104.0",
        "jinja2>=3.1.0"
    ])
    .pip_install("git+https://github.com/openai/CLIP.git")
)

app = modal.App("kure-web-interface")

# Volumes for persistent storage
outputs_volume = modal.Volume.from_name("kure-outputs", create_if_missing=True)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KURE Project - AI Image Generation</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }
        .header h1 {
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        .form-container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        input[type="text"], input[type="number"], select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus, input[type="number"]:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        .form-row {
            display: flex;
            gap: 20px;
        }
        .form-row .form-group {
            flex: 1;
        }
        .generate-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 18px;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
            transition: transform 0.2s;
        }
        .generate-btn:hover {
            transform: translateY(-2px);
        }
        .generate-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: white;
            font-size: 18px;
        }
        .results {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .image-item {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        .image-item:hover {
            transform: scale(1.05);
        }
        .image-item img {
            width: 100%;
            height: auto;
            display: block;
        }
        .error {
            background: #ff6b6b;
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .success {
            background: #51cf66;
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 KURE Project</h1>
            <p>Nonlinear Rectified Flows for AI Image Generation</p>
        </div>
        
        <div class="form-container">
            <form id="generateForm">
                <div class="form-group">
                    <label for="prompt">Image Description:</label>
                    <input type="text" id="prompt" name="prompt" 
                           placeholder="Describe the image you want to generate..." required>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="steps">Sampling Steps:</label>
                        <select id="steps" name="steps">
                            <option value="1">1 (Fastest)</option>
                            <option value="2">2</option>
                            <option value="4" selected>4 (Recommended)</option>
                            <option value="8">8</option>
                            <option value="16">16 (Highest Quality)</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="num_samples">Number of Images:</label>
                        <select id="num_samples" name="num_samples">
                            <option value="1">1</option>
                            <option value="4">4</option>
                            <option value="8" selected>8</option>
                            <option value="16">16</option>
                        </select>
                    </div>
                </div>
                
                <button type="submit" class="generate-btn" id="generateBtn">
                    🚀 Generate Images
                </button>
            </form>
        </div>
        
        <div id="loading" class="loading" style="display: none;">
            <p>🎨 Generating your images... This may take a few minutes.</p>
        </div>
        
        <div id="results" class="results" style="display: none;">
            <h2>Generated Images</h2>
            <div id="imageGrid" class="image-grid"></div>
        </div>
        
        <div id="error" class="error" style="display: none;"></div>
    </div>

    <script>
        document.getElementById('generateForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());
            
            // Show loading, hide results
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            document.getElementById('generateBtn').disabled = true;
            document.getElementById('generateBtn').textContent = '🎨 Generating...';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    displayResults(result);
                } else {
                    showError('Generation failed: ' + (result.error || 'Unknown error'));
                }
            } catch (error) {
                showError('Network error: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('generateBtn').disabled = false;
                document.getElementById('generateBtn').textContent = '🚀 Generate Images';
            }
        });
        
        function displayResults(result) {
            const resultsDiv = document.getElementById('results');
            const imageGrid = document.getElementById('imageGrid');
            
            imageGrid.innerHTML = '';
            
            if (result.images && result.images.length > 0) {
                result.images.forEach((imageData, index) => {
                    const imageItem = document.createElement('div');
                    imageItem.className = 'image-item';
                    
                    const img = document.createElement('img');
                    img.src = 'data:image/png;base64,' + imageData;
                    img.alt = `Generated image ${index + 1}`;
                    
                    imageItem.appendChild(img);
                    imageGrid.appendChild(imageItem);
                });
                
                resultsDiv.style.display = 'block';
            } else {
                showError('No images were generated');
            }
        }
        
        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        }
    </script>
</body>
</html>
"""

@app.function(
    image=image,
    gpu=modal.gpu.T4(),
    volumes={"/outputs": outputs_volume},
    timeout=600
)
def generate_images_for_web(prompt: str, steps: int = 4, num_samples: int = 8):
    """Generate images and return as base64 encoded strings"""
    import subprocess
    import sys
    import os
    from datetime import datetime
    import base64
    
    # Clone the repository if not exists
    if not os.path.exists("/tmp/KURE-Project"):
        subprocess.run([
            "git", "clone", 
            "https://github.com/Todd7777/KURE-Project.git",
            "/tmp/KURE-Project"
        ], check=True)
    
    os.chdir("/tmp/KURE-Project")
    
    # Install the package
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_dir = f"/outputs/web_samples_{timestamp}"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Run sampling (simplified version for web interface)
    cmd = [
        sys.executable, "scripts/sample.py",
        "--prompt", prompt,
        "--steps", str(steps),
        "--num_samples", str(num_samples),
        "--output_dir", sample_dir
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Sampling failed: {result.stderr}")
    
    # Convert images to base64
    images = []
    if os.path.exists(sample_dir):
        for filename in os.listdir(sample_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(sample_dir, filename)
                with open(filepath, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                    images.append(image_data)
    
    return {
        "status": "success",
        "images": images,
        "prompt": prompt,
        "steps": steps,
        "num_samples": num_samples
    }

@app.function(image=image)
@web_endpoint(method="GET")
def web_interface():
    """Serve the web interface"""
    return modal.Response(
        content=HTML_TEMPLATE,
        headers={"Content-Type": "text/html"}
    )

@app.function(image=image, gpu=modal.gpu.T4(), volumes={"/outputs": outputs_volume})
@web_endpoint(method="POST")
def generate_endpoint(request_data: dict):
    """API endpoint for image generation"""
    try:
        prompt = request_data.get("prompt", "")
        steps = int(request_data.get("steps", 4))
        num_samples = int(request_data.get("num_samples", 8))
        
        if not prompt:
            return {"status": "error", "error": "Prompt is required"}
        
        # Limit parameters for web interface
        steps = max(1, min(16, steps))
        num_samples = max(1, min(16, num_samples))
        
        result = generate_images_for_web.remote(prompt, steps, num_samples)
        return result
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    # Deploy the web interface
    with app.run():
        print("🌐 Web interface is running!")
        print("Visit the URL provided by Modal to access the interface.")
