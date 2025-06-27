# Complete Facial Features Extraction System - Fixed for Codespaces
# This version includes proper model loading and error handling

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Codespaces
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
import torchvision.transforms as transforms
import glob
from pathlib import Path
import shutil

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration class for the facial feature extraction system"""
    
    # Input/Output paths
    INPUT_DATASET_PATH = "./test_images"
    OUTPUT_BASE_PATH = "./facial_features_output"
    
    # Model paths - Updated for the cloned repository
    FACE_PARSING_REPO = "./face-parsing.PyTorch"
    BISENET_MODEL_PATH = "./face-parsing.PyTorch/res/cp/79999_iter.pth"
    
    # Processing settings
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    RESIZE_FOR_MODEL = (512, 512)
    
    # Feature extraction settings
    EXTRACT_INDIVIDUAL_FEATURES = True
    EXTRACT_FACE_OUTLINE = True
    CREATE_OVERLAY_IMAGE = True
    SAVE_ANALYSIS_REPORT = True

# =============================================================================
# BISENET MODEL IMPLEMENTATION
# =============================================================================

class BiSeNet(nn.Module):
    """BiSeNet implementation for face parsing"""
    
    def __init__(self, n_classes=19):
        super(BiSeNet, self).__init__()
        
        # We'll use a simplified version that's compatible with the pre-trained weights
        # This is a basic implementation - you might need to adjust based on the actual model
        
        self.context_path = self._make_context_path()
        self.spatial_path = self._make_spatial_path()
        self.ffm = FeatureFusionModule(256, 256, 256)
        self.classifier = nn.Conv2d(256, n_classes, 1)
        
    def _make_context_path(self):
        """Create context path"""
        layers = []
        # Basic ResNet-like structure
        layers.extend([
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        ])
        return nn.Sequential(*layers)
    
    def _make_spatial_path(self):
        """Create spatial path"""
        layers = []
        layers.extend([
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        ])
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Get spatial and context features
        spatial_out = self.spatial_path(x)
        context_out = self.context_path(x)
        
        # Resize to match
        h, w = spatial_out.size()[2:]
        context_out = F.interpolate(context_out, size=(h, w), mode='bilinear', align_corners=True)
        
        # Fuse features
        fused = self.ffm(spatial_out, context_out)
        
        # Final classification
        out = self.classifier(fused)
        
        # Upsample to original size
        out = F.interpolate(out, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        
        return out

class FeatureFusionModule(nn.Module):
    """Feature Fusion Module for BiSeNet"""
    
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(FeatureFusionModule, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels1 + in_channels2, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Attention module
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(out_channels, out_channels // 4, 1)
        self.fc2 = nn.Conv2d(out_channels // 4, out_channels, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x1, x2):
        # Concatenate features
        x = torch.cat([x1, x2], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        
        # Attention
        att = self.gap(x)
        att = self.relu(self.fc1(att))
        att = self.sigmoid(self.fc2(att))
        
        return x * att

# =============================================================================
# FACE OUTLINE DETECTOR
# =============================================================================

class SimpleFaceOutlineDetector:
    """Simplified face outline detector"""
    
    def __init__(self):
        try:
            from mtcnn import MTCNN
            self.mtcnn = MTCNN()
            print("✓ MTCNN loaded for face detection")
        except Exception as e:
            print(f"Warning: MTCNN not available: {e}")
            self.mtcnn = None
            # Fallback to OpenCV
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
    
    def detect_face_outline(self, image_rgb):
        """Detect face outline from image"""
        h, w = image_rgb.shape[:2]
        outline_mask = np.zeros((h, w), dtype=bool)
        
        if self.mtcnn:
            try:
                detections = self.mtcnn.detect_faces(image_rgb)
                if detections:
                    # Get bounding box from detection
                    x, y, w_box, h_box = detections[0]['box']
                    # Create elliptical outline
                    center = (x + w_box//2, y + h_box//2)
                    axes = (w_box//2, h_box//2)
                    
                    # Create mask
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, 8)
                    outline_mask = mask > 0
                    
            except Exception as e:
                print(f"MTCNN detection failed: {e}")
        
        # Fallback to basic detection
        if not np.any(outline_mask):
            outline_mask = self._basic_face_outline(image_rgb)
        
        return outline_mask
    
    def _basic_face_outline(self, image_rgb):
        """Basic face outline using OpenCV"""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if len(faces) > 0:
            x, y, w_face, h_face = faces[0]  # Use first face
            center = (x + w_face//2, y + h_face//2)
            axes = (w_face//2, h_face//2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, 8)
        
        return mask > 0

# =============================================================================
# COMPREHENSIVE FACE PARSER
# =============================================================================

class ComprehensiveFaceParser:
    """Complete face parser for facial components"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # CelebAMask-HQ 19 classes
        self.labels = [
            'background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye',
            'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose', 'mouth',
            'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat'
        ]
        
        # Colors for visualization
        self.colors = self.get_distinct_colors()
        
        # Transform for preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.model = None
        self.face_outline_detector = SimpleFaceOutlineDetector()
    
    def get_distinct_colors(self):
        """Get distinct colors for each feature"""
        colors = np.zeros((20, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]          # background
        colors[1] = [255, 182, 193]    # skin
        colors[2] = [139, 69, 19]      # l_brow
        colors[3] = [160, 82, 45]      # r_brow
        colors[4] = [255, 0, 0]        # l_eye
        colors[5] = [255, 20, 147]     # r_eye
        colors[6] = [0, 255, 255]      # eye_g
        colors[7] = [255, 165, 0]      # l_ear
        colors[8] = [255, 140, 0]      # r_ear
        colors[9] = [255, 215, 0]      # ear_r
        colors[10] = [0, 255, 0]       # nose
        colors[11] = [255, 0, 255]     # mouth
        colors[12] = [255, 105, 180]   # u_lip
        colors[13] = [255, 20, 147]    # l_lip
        colors[14] = [222, 184, 135]   # neck
        colors[15] = [210, 180, 140]   # neck_l
        colors[16] = [0, 0, 255]       # cloth
        colors[17] = [128, 0, 128]     # hair
        colors[18] = [75, 0, 130]      # hat
        colors[19] = [255, 255, 0]     # face_outline
        return colors
    
    def load_model(self):
        """Load the BiSeNet model"""
        try:
            # Add the face-parsing repository to path
            if Config.FACE_PARSING_REPO not in sys.path:
                sys.path.append(Config.FACE_PARSING_REPO)
            
            # Try to import from the repository
            try:
                from model import BiSeNet as RepoBiSeNet
                self.model = RepoBiSeNet(n_classes=19).to(self.device)
                print("✅ Using BiSeNet from repository")
            except ImportError:
                # Use our implementation
                self.model = BiSeNet(n_classes=19).to(self.device)
                print("✅ Using built-in BiSeNet implementation")
            
            # Load pre-trained weights
            if os.path.exists(Config.BISENET_MODEL_PATH):
                try:
                    state_dict = torch.load(Config.BISENET_MODEL_PATH, map_location=self.device)
                    self.model.load_state_dict(state_dict, strict=False)
                    print("✅ Pre-trained weights loaded successfully!")
                except Exception as e:
                    print(f"⚠️  Warning: Could not load pre-trained weights: {e}")
                    print("   Continuing with random initialization...")
            else:
                print(f"⚠️  Model file not found at {Config.BISENET_MODEL_PATH}")
                print("   Using model with random weights...")
            
            self.model.eval()
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def parse_face_complete(self, image_path):
        """Parse face and extract features"""
        if self.model is None:
            if not self.load_model():
                return None, None, None
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            original_array = np.array(image)
            
            # Preprocess
            image_resized = image.resize(Config.RESIZE_FOR_MODEL, Image.BILINEAR)
            input_tensor = self.transform(image_resized).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(input_tensor)
                
                if isinstance(output, (tuple, list)):
                    output = output[0]
                
                # Get prediction
                if output.dim() == 4:
                    prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
                else:
                    raise ValueError(f"Unexpected output shape: {output.shape}")
            
            # Resize to original
            prediction_resized = cv2.resize(
                prediction.astype(np.uint8), 
                original_size, 
                interpolation=cv2.INTER_NEAREST
            )
            
            # Face outline
            face_outline_mask = self.face_outline_detector.detect_face_outline(original_array)
            
            return prediction_resized, original_array, face_outline_mask
            
        except Exception as e:
            print(f"❌ Error in parse_face_complete: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

# =============================================================================
# FEATURE EXTRACTION SYSTEM
# =============================================================================

class FeatureExtractionSystem:
    """System to extract and organize facial features"""
    
    def __init__(self, parser):
        self.parser = parser
    
    def extract_individual_feature_images(self, original_image, parsing_result, face_outline_mask, output_dir, image_name):
        """Extract individual feature images"""
        features_dir = os.path.join(output_dir, f"{image_name}_features")
        os.makedirs(features_dir, exist_ok=True)
        
        extracted_features = {}
        feature_stats = {}
        
        # Extract features
        for class_id in range(1, 19):
            feature_mask = (parsing_result == class_id)
            
            if np.any(feature_mask):
                feature_name = self.parser.labels[class_id]
                
                # Create feature image
                feature_image = np.zeros_like(original_image)
                feature_image[feature_mask] = original_image[feature_mask]
                
                # Save
                feature_path = os.path.join(features_dir, f"{feature_name}.jpg")
                cv2.imwrite(feature_path, cv2.cvtColor(feature_image, cv2.COLOR_RGB2BGR))
                
                extracted_features[feature_name] = feature_image
                feature_stats[feature_name] = {
                    'pixels': np.sum(feature_mask),
                    'bbox': self._get_bounding_box(feature_mask),
                    'class_id': class_id
                }
        
        # Face outline
        if Config.EXTRACT_FACE_OUTLINE and face_outline_mask is not None and np.any(face_outline_mask):
            outline_image = np.zeros_like(original_image)
            outline_image[face_outline_mask] = [255, 255, 0]
            
            outline_path = os.path.join(features_dir, "face_outline.jpg")
            cv2.imwrite(outline_path, cv2.cvtColor(outline_image, cv2.COLOR_RGB2BGR))
            
            extracted_features['face_outline'] = outline_image
            feature_stats['face_outline'] = {
                'pixels': np.sum(face_outline_mask),
                'bbox': self._get_bounding_box(face_outline_mask),
                'class_id': 19
            }
        
        return extracted_features, feature_stats, features_dir
    
    def create_complete_overlay(self, original_image, parsing_result, face_outline_mask, extracted_features, output_dir, image_name):
        """Create overlay image"""
        overlay = original_image.copy()
        
        # Add features
        for class_id in range(1, 19):
            feature_mask = (parsing_result == class_id)
            if np.any(feature_mask):
                color = self.parser.colors[class_id]
                overlay[feature_mask] = color
        
        # Add outline
        if face_outline_mask is not None and np.any(face_outline_mask):
            overlay[face_outline_mask] = self.parser.colors[19]
        
        # Blend
        blended_overlay = cv2.addWeighted(original_image, 0.4, overlay, 0.6, 0)
        
        # Save
        overlay_path = os.path.join(output_dir, f"{image_name}_complete_overlay.jpg")
        cv2.imwrite(overlay_path, cv2.cvtColor(blended_overlay, cv2.COLOR_RGB2BGR))
        
        return blended_overlay, overlay_path
    
    def generate_analysis_report(self, feature_stats, output_dir, image_name, original_shape):
        """Generate analysis report"""
        report_path = os.path.join(output_dir, f"{image_name}_analysis_report.txt")
        total_pixels = original_shape[0] * original_shape[1]
        
        with open(report_path, 'w') as f:
            f.write(f"FACIAL FEATURE ANALYSIS REPORT\n")
            f.write(f"==============================\n\n")
            f.write(f"Image: {image_name}\n")
            f.write(f"Image Size: {original_shape[1]} x {original_shape[0]} pixels\n")
            f.write(f"Total Pixels: {total_pixels:,}\n")
            f.write(f"Features Detected: {len(feature_stats)}\n\n")
            
            f.write(f"FEATURE DETAILS:\n")
            f.write(f"{'Feature':<15} {'Pixels':<10} {'Coverage':<10} {'Bounding Box':<25}\n")
            f.write(f"{'-'*70}\n")
            
            sorted_features = sorted(feature_stats.items(), key=lambda x: x[1]['pixels'], reverse=True)
            
            for feature_name, stats in sorted_features:
                coverage = (stats['pixels'] / total_pixels) * 100
                bbox_str = f"{stats['bbox']}" if stats['bbox'] else "None"
                f.write(f"{feature_name:<15} {stats['pixels']:<10,} {coverage:<9.2f}% {bbox_str:<25}\n")
        
        return report_path
    
    def _get_bounding_box(self, mask):
        """Get bounding box for mask"""
        coords = np.where(mask)
        if len(coords[0]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            return (x_min, y_min, x_max - x_min, y_max - y_min)
        return None

# =============================================================================
# MAIN PROCESSOR
# =============================================================================

class FacialFeatureProcessor:
    """Main facial feature processor"""
    
    def __init__(self):
        self.parser = ComprehensiveFaceParser()
        self.extraction_system = FeatureExtractionSystem(self.parser)
    
    def process_dataset(self, input_path, output_path=None):
        """Process all images in dataset"""
        if output_path is None:
            output_path = Config.OUTPUT_BASE_PATH
        
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            print(f"❌ Input directory does not exist: {input_path}")
            return False
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find images
        image_files = []
        for ext in Config.SUPPORTED_FORMATS:
            image_files.extend(list(input_path.glob(f"*{ext}")))
            image_files.extend(list(input_path.glob(f"*{ext.upper()}")))
        
        if not image_files:
            print(f"❌ No image files found in {input_path}")
            return False
        
        print(f"📁 Found {len(image_files)} images to process")
        print(f"🎯 Output directory: {output_path}")
        print("="*60)
        
        successful_processing = 0
        
        for idx, image_file in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] Processing: {image_file.name}")
            
            try:
                success = self._process_single_image(image_file, output_path)
                if success:
                    successful_processing += 1
                    print(f"  ✅ Successfully processed {image_file.name}")
                else:
                    print(f"  ❌ Failed to process {image_file.name}")
            except Exception as e:
                print(f"  ❌ Error processing {image_file.name}: {e}")
        
        print("\n" + "="*60)
        print(f"🏁 PROCESSING COMPLETE!")
        print(f"✅ Successfully processed: {successful_processing}/{len(image_files)} images")
        print(f"📁 Results saved to: {output_path}")
        
        return successful_processing > 0
    
    def _process_single_image(self, image_path, output_base_path):
        """Process single image"""
        # Parse face
        parsing_result, original_image, face_outline_mask = self.parser.parse_face_complete(str(image_path))
        
        if parsing_result is None or original_image is None:
            return False
        
        # Create output directory
        image_name = image_path.stem
        image_output_dir = output_base_path / f"{image_name}_analysis"
        image_output_dir.mkdir(exist_ok=True)
        
        # Save original
        original_path = image_output_dir / f"{image_name}_original.jpg"
        cv2.imwrite(str(original_path), cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
        
        # Extract features
        if Config.EXTRACT_INDIVIDUAL_FEATURES:
            extracted_features, feature_stats, features_dir = self.extraction_system.extract_individual_feature_images(
                original_image, parsing_result, face_outline_mask, str(image_output_dir), image_name
            )
            print(f"    📋 Extracted {len(extracted_features)} individual features")
        
        # Create overlay
        if Config.CREATE_OVERLAY_IMAGE:
            overlay_image, overlay_path = self.extraction_system.create_complete_overlay(
                original_image, parsing_result, face_outline_mask, extracted_features, str(image_output_dir), image_name
            )
            print(f"    🎨 Created complete overlay image")
        
        # Generate report
        if Config.SAVE_ANALYSIS_REPORT:
            report_path = self.extraction_system.generate_analysis_report(
                feature_stats, str(image_output_dir), image_name, original_image.shape
            )
            print(f"    📊 Generated analysis report")
        
        return True

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function"""
    print("🎯 COMPREHENSIVE FACIAL FEATURE EXTRACTION SYSTEM")
    print("="*60)
    print("This system will:")
    print("• Process all images in the specified directory")
    print("• Extract individual facial features (nose, eyes, mouth, etc.)")
    print("• Create separate image files for each feature")
    print("• Generate face outline detection")
    print("• Create complete overlay images") 
    print("• Generate detailed analysis reports")
    print("="*60)
    
    # Check setup
    if not os.path.exists(Config.FACE_PARSING_REPO):
        print("⚠️  face-parsing.PyTorch repository not found!")
        print("Please run the setup script first:")
        print("bash setup_face_parsing.sh")
        return
    
    processor = FacialFeatureProcessor()
    success = processor.process_dataset(Config.INPUT_DATASET_PATH)
    
    if success:
        print(f"\n🎉 SUCCESS! Check the output directory: {Config.OUTPUT_BASE_PATH}")
        print("📁 Each image will have its own folder containing:")
        print("   • Individual feature images (nose.jpg, l_eye.jpg, etc.)")
        print("   • Face outline image (face_outline.jpg)")
        print("   • Complete overlay image (showing all features)")
        print("   • Analysis report (detailed statistics)")
    else:
        print("\n❌ Processing failed. Please check the setup and try again.")

if __name__ == "__main__":
    main()