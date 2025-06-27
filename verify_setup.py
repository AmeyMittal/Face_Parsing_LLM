# Quick setup verification script
import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path

def verify_setup():
    """Verify that all components are properly set up"""
    
    print("🔍 VERIFYING FACE PARSING SETUP")
    print("="*50)
    
    checks_passed = 0
    total_checks = 6
    
    # Check 1: OpenCV
    try:
        print(f"1. OpenCV version: {cv2.__version__}")
        checks_passed += 1
    except Exception as e:
        print(f"1. ❌ OpenCV issue: {e}")
    
    # Check 2: PyTorch
    try:
        print(f"2. PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        checks_passed += 1
    except Exception as e:
        print(f"2. ❌ PyTorch issue: {e}")
    
    # Check 3: face-parsing repository
    repo_path = "face-parsing.PyTorch"
    if os.path.exists(repo_path):
        print(f"3. ✅ face-parsing.PyTorch repository found")
        checks_passed += 1
    else:
        print(f"3. ❌ face-parsing.PyTorch repository NOT found")
    
    # Check 4: Model file
    model_path = "face-parsing.PyTorch/res/cp/79999_iter.pth"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024*1024)
        print(f"4. ✅ Model file found ({size_mb:.1f} MB)")
        checks_passed += 1
    else:
        print(f"4. ❌ Model file NOT found at {model_path}")
    
    # Check 5: MTCNN
    try:
        from mtcnn import MTCNN
        print("5. ✅ MTCNN available")
        checks_passed += 1
    except Exception as e:
        print(f"5. ⚠️  MTCNN not available: {e}")
        print("   (Will use OpenCV fallback)")
    
    # Check 6: Directories
    test_dir = Path("test_images")
    output_dir = Path("facial_features_output")
    
    if test_dir.exists() and output_dir.exists():
        print("6. ✅ Directories created")
        
        # Count test images
        image_count = len(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")))
        print(f"   Test images found: {image_count}")
        checks_passed += 1
    else:
        print("6. ❌ Directories missing")
    
    print(f"\n📊 SETUP STATUS: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed >= 4:  # Minimum required
        print("✅ Setup looks good! You can run the face parsing script.")
        
        if checks_passed < total_checks:
            print("\n💡 Suggestions to improve setup:")
            if not os.path.exists(model_path):
                print("   - Download the BiSeNet model for better results")
            if checks_passed < 5:
                print("   - Install MTCNN: pip install mtcnn")
    else:
        print("❌ Setup incomplete. Please resolve the issues above.")
    
    return checks_passed >= 4

def create_sample_image():
    """Create a sample face image for testing"""
    # Create a simple synthetic face-like image
    img = np.ones((300, 300, 3), dtype=np.uint8) * 220  # Light background
    
    # Add simple face-like features
    # Face oval
    cv2.ellipse(img, (150, 150), (80, 100), 0, 0, 360, (255, 220, 177), -1)
    
    # Eyes
    cv2.circle(img, (130, 130), 8, (0, 0, 0), -1)  # Left eye
    cv2.circle(img, (170, 130), 8, (0, 0, 0), -1)  # Right eye
    
    # Nose
    cv2.circle(img, (150, 150), 3, (200, 180, 150), -1)
    
    # Mouth
    cv2.ellipse(img, (150, 170), (15, 8), 0, 0, 180, (150, 50, 50), 2)
    
    # Save sample image
    os.makedirs("test_images", exist_ok=True)
    cv2.imwrite("test_images/sample_face.jpg", img)
    print("📷 Created sample face image: test_images/sample_face.jpg")

if __name__ == "__main__":
    if verify_setup():
        # Create a sample image if no test images exist
        test_dir = Path("test_images")
        if not any(test_dir.glob("*.jpg")) and not any(test_dir.glob("*.png")):
            create_sample_image()
        
        print("\n🚀 Ready to run face parsing!")
        print("   python working_face_parser.py")
    else:
        print("\n🔧 Please complete the setup first.")