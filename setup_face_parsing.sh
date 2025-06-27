#!/bin/bash

echo "🚀 Setting up Face Parsing System in GitHub Codespaces"
echo "======================================================"

# 1. Clone the face-parsing repository
echo "📥 Cloning face-parsing.PyTorch repository..."
git clone https://github.com/zllrunning/face-parsing.PyTorch.git

# 2. Download the pre-trained model
echo "📥 Downloading BiSeNet pre-trained model..."
cd face-parsing.PyTorch
mkdir -p res/cp
cd res/cp

# Download the model (this might take a few minutes)
wget -O 79999_iter.pth "https://drive.google.com/uc?export=download&id=154JgKpzCPW82qINcVieuPH3fZ2e0P812"

# Alternative download method if wget fails
if [ ! -f "79999_iter.pth" ]; then
    echo "⚠️  Direct download failed. Trying alternative method..."
    curl -L "https://drive.google.com/uc?export=download&id=154JgKpzCPW82qINcVieuPH3fZ2e0P812" -o 79999_iter.pth
fi

# Check if download was successful
if [ -f "79999_iter.pth" ]; then
    echo "✅ Model downloaded successfully!"
    ls -lh 79999_iter.pth
else
    echo "❌ Model download failed. You may need to download manually."
fi

cd ../../..

# 3. Install additional requirements
echo "📦 Installing additional Python packages..."
pip install opencv-python-headless
pip install mtcnn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib pillow numpy

# 4. Create test images directory
echo "📁 Setting up directories..."
mkdir -p test_images
mkdir -p facial_features_output

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Add some test images to the test_images/ directory"
echo "2. Run the updated face parsing script"
echo ""
echo "🔍 Checking setup..."
echo "face-parsing.PyTorch directory: $(ls -la face-parsing.PyTorch/)"
echo "Model file: $(ls -la face-parsing.PyTorch/res/cp/ 2>/dev/null || echo 'Model not found')"