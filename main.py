# Complete imports and setup for Google Colab
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import random
import json
from PIL import Image
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Check if running in Colab and install required packages
try:
    import google.colab
    IN_COLAB = True
    print("Running in Google Colab")
    # Install required packages if not already installed
    import subprocess
    import sys

    def install_package(package):
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    # Install required packages
    install_package('opencv-python-headless')
    install_package('scikit-learn')
    install_package('matplotlib')
    install_package('seaborn')

except ImportError:
    IN_COLAB = False
    print("Not running in Google Colab")

# Import required packages
try:
    import cv2
    print("✓ OpenCV imported successfully")
except ImportError as e:
    print(f"✗ OpenCV import failed: {e}")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    print("✓ Scikit-learn imported successfully")
except ImportError as e:
    print(f"✗ Scikit-learn import failed: {e}")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("✓ Matplotlib and Seaborn imported successfully")
except ImportError as e:
    print(f"✗ Matplotlib/Seaborn import failed: {e}")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("CUDA not available, using CPU")

# Set matplotlib style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300


# Utility classes
class ContrastiveLoss(nn.Module):
    """Contrastive loss for Siamese networks"""
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

class VGG16FeatureExtractor(nn.Module):
    """VGG16-based feature extractor with improved error handling"""
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        try:
            print("Loading VGG16 pretrained model...")
            vgg16 = models.vgg16(pretrained=True)
            print("✓ VGG16 loaded successfully")

            self.features = vgg16.features

            # Create adaptive classifier to handle different input sizes
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 1024),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(1024, 512)
            )
            print("✓ Feature extractor initialized")

        except Exception as e:
            print(f"✗ Error initializing VGG16: {e}")
            raise

    def forward(self, x):
        try:
            x = self.features(x)
            x = self.classifier(x)
            return x
        except Exception as e:
            print(f"Error in VGG16 forward pass: {e}")
            print(f"Input shape: {x.shape}")
            raise

class SiameseNetwork(nn.Module):
    """Siamese network for feature embedding"""
    def __init__(self, input_dim=512, embedding_dim=512):
        super(SiameseNetwork, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
        )

    def forward_one(self, x):
        output = self.embedding(x)
        # L2 normalize using functional API
        return F.normalize(output, p=2, dim=1)

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

# Debug version of MultiModalCorrosionAnalyzer
class MultiModalCorrosionAnalyzer:
    """
    Debug version with verbose logging for Google Colab
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        print(f"Initializing MultiModalCorrosionAnalyzer on {device}")
        self.device = device

        # Initialize VGG16 feature extractors for each modality
        print("Loading VGG16 feature extractors...")
        self.rgb_extractor = VGG16FeatureExtractor().to(device)
        self.thermal_extractor = VGG16FeatureExtractor().to(device)
        self.lidar_extractor = VGG16FeatureExtractor().to(device)

        # Initialize Siamese networks for each modality
        print("Initializing Siamese networks...")
        self.rgb_siamese = SiameseNetwork().to(device)
        self.thermal_siamese = SiameseNetwork().to(device)
        self.lidar_siamese = SiameseNetwork().to(device)

        # Fusion network
        print("Building fusion network...")
        self.fusion_net = self._build_fusion_network().to(device)

        # Preprocessing transforms
        self.rgb_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        self.thermal_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        self.lidar_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        print("MultiModalCorrosionAnalyzer initialized successfully!")

    def _build_fusion_network(self):
        """Build multi-modal fusion network"""
        return nn.Sequential(
            nn.Linear(3 * 512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

    def check_dataset_structure(self, data_directory):
        """Check and print dataset structure"""
        print(f"\nChecking dataset structure in: {data_directory}")

        if not os.path.exists(data_directory):
            print(f"ERROR: Directory {data_directory} does not exist!")
            return False

        total_files = 0
        for root, dirs, files in os.walk(data_directory):
            level = root.replace(data_directory, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            png_files = [f for f in files if f.endswith('.png')]
            for file in png_files[:5]:  # Show first 5 files
                print(f"{subindent}{file}")
                total_files += 1
            if len(png_files) > 5:
                print(f"{subindent}... and {len(png_files) - 5} more files")
                total_files += len(png_files) - 5

        print(f"\nTotal PNG files found: {total_files}")
        return total_files > 0



    def prepare_training_dataset(self, data_directory, validation_split=0.2):
        """Prepare training dataset from directory structure with detailed logging"""
        print(f"\nPreparing training dataset from: {data_directory}")

        dataset = defaultdict(list)
        total_samples = 0

        # Scan directory for samples
        for class_dir in os.listdir(data_directory):
            class_path = os.path.join(data_directory, class_dir)
            if not os.path.isdir(class_path):
                continue

            print(f"Processing class directory: {class_dir}")

            # Extract class label from directory name
            if 'no_corrosion' in class_dir.lower():
                class_label = 0
            elif 'mild' in class_dir.lower():
                class_label = 1
            elif 'severe' in class_dir.lower():
                class_label = 2
            else:
                try:
                    class_label = int(class_dir.split('_')[1])
                except:
                    class_label = 0

            # Group files by sample ID - handle nested structure
            samples = defaultdict(dict)
            
            # Check if there are subdirectories (sample_1, sample_2, etc.)
            subdirs = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]
            
            if subdirs:
                # Handle nested structure: class_dir/sample_X/files
                print(f"  Found {len(subdirs)} sample subdirectories")
                for subdir in subdirs:
                    subdir_path = os.path.join(class_path, subdir)
                    files = [f for f in os.listdir(subdir_path) if f.endswith('.png')]
                    
                    for file in files:
                        # Handle different naming conventions
                        if '_therm.png' in file:
                            # Thermal image
                            base_name = file.replace('_therm.png', '')
                            samples[base_name]['thermal'] = os.path.join(subdir_path, file)
                        elif '_depth.png' in file:
                            # LiDAR depth image
                            base_name = file.replace('_depth.png', '')
                            samples[base_name]['lidar'] = os.path.join(subdir_path, file)
                        elif file.endswith('.png') and not ('_therm' in file or '_depth' in file):
                            # RGB image (default)
                            base_name = file.replace('.png', '')
                            samples[base_name]['rgb'] = os.path.join(subdir_path, file)
            else:
                # Handle flat structure: class_dir/files
                files = [f for f in os.listdir(class_path) if f.endswith('.png')]
                print(f"  Found {len(files)} PNG files in flat structure")
                
                for file in files:
                    # Handle different naming conventions
                    if '_therm.png' in file:
                        # Thermal image
                        base_name = file.replace('_therm.png', '')
                        samples[base_name]['thermal'] = os.path.join(class_path, file)
                    elif '_depth.png' in file:
                        # LiDAR depth image
                        base_name = file.replace('_depth.png', '')
                        samples[base_name]['lidar'] = os.path.join(class_path, file)
                    elif file.endswith('.png') and not ('_therm' in file or '_depth' in file):
                        # RGB image (default)
                        base_name = file.replace('.png', '')
                        samples[base_name]['rgb'] = os.path.join(class_path, file)

            # Add complete samples to dataset
            class_count = 0
            for sample_id, modalities in samples.items():
                if len(modalities) > 0:
                    dataset[class_label].append({
                        'sample_id': sample_id,
                        'class': class_label,
                        **modalities
                    })
                    class_count += 1

            print(f"  Added {class_count} samples for class {class_label}")
            total_samples += class_count

        print(f"\nTotal samples found: {total_samples}")

        if total_samples == 0:
            print("ERROR: No samples found in the dataset!")
            return [], []

        # Split into train/validation
        train_data, val_data = [], []
        for class_label, samples in dataset.items():
            random.shuffle(samples)
            split_idx = int(len(samples) * (1 - validation_split))
            train_data.extend(samples[:split_idx])
            val_data.extend(samples[split_idx:])
            print(f"Class {class_label}: {split_idx} train, {len(samples) - split_idx} val")

        print(f"\nFinal split: {len(train_data)} train, {len(val_data)} validation")
        return train_data, val_data

    def process_lidar_depth_image(self, lidar_path):
        """Process LiDAR depth map image (.png format)"""
        try:
            depth_image = cv2.imread(lidar_path, cv2.IMREAD_UNCHANGED)

            if depth_image is None:
                raise ValueError(f"Could not load image: {lidar_path}")

            # Handle different bit depths
            if depth_image.dtype == np.uint16:
                depth_norm = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
                depth_8bit = depth_norm.astype(np.uint8)
            elif depth_image.dtype == np.uint8:
                depth_8bit = depth_image
            else:
                depth_norm = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
                depth_8bit = depth_norm.astype(np.uint8)

            # Convert to RGB
            if len(depth_8bit.shape) == 2:
                depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
                depth_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
            else:
                if depth_8bit.shape[2] == 3:
                    depth_rgb = cv2.cvtColor(depth_8bit, cv2.COLOR_BGR2RGB)
                else:
                    depth_rgb = depth_8bit

            return Image.fromarray(depth_rgb)

        except Exception as e:
            print(f"Error processing LiDAR depth image {lidar_path}: {e}")
            # Return synthetic depth image
            synthetic_depth = self._generate_synthetic_depth((224, 224))
            return Image.fromarray(synthetic_depth)

    def process_thermal_image(self, thermal_path):
        """Process thermal image (.png format)"""
        try:
            thermal_image = cv2.imread(thermal_path, cv2.IMREAD_UNCHANGED)

            if thermal_image is None:
                raise ValueError(f"Could not load image: {thermal_path}")

            # Handle different formats
            if thermal_image.dtype == np.uint16:
                thermal_norm = cv2.normalize(thermal_image, None, 0, 255, cv2.NORM_MINMAX)
                thermal_8bit = thermal_norm.astype(np.uint8)
            elif thermal_image.dtype == np.uint8:
                thermal_8bit = thermal_image
            else:
                thermal_norm = cv2.normalize(thermal_image, None, 0, 255, cv2.NORM_MINMAX)
                thermal_8bit = thermal_norm.astype(np.uint8)

            # Convert to RGB
            if len(thermal_8bit.shape) == 2:
                thermal_colored = cv2.applyColorMap(thermal_8bit, cv2.COLORMAP_INFERNO)
                thermal_rgb = cv2.cvtColor(thermal_colored, cv2.COLOR_BGR2RGB)
            elif thermal_8bit.shape[2] == 3:
                thermal_rgb = cv2.cvtColor(thermal_8bit, cv2.COLOR_BGR2RGB)
            else:
                thermal_rgb = thermal_8bit

            return Image.fromarray(thermal_rgb)

        except Exception as e:
            print(f"Error processing thermal image {thermal_path}: {e}")
            synthetic = self._generate_synthetic_thermal((224, 224))
            return Image.fromarray(synthetic)

    def _generate_synthetic_depth(self, image_size):
        """Generate synthetic depth image"""
        depth = np.ones(image_size, dtype=np.float32) * 0.5
        num_pits = np.random.randint(5, 20)
        for _ in range(num_pits):
            center_x = np.random.randint(20, image_size[1] - 20)
            center_y = np.random.randint(20, image_size[0] - 20)
            radius = np.random.randint(5, 15)
            pit_depth = np.random.uniform(0.1, 0.9)

            y, x = np.ogrid[:image_size[0], :image_size[1]]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
            depth[mask] = pit_depth

        depth_norm = (depth * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        return cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

    def _generate_synthetic_thermal(self, image_size):
        """Generate synthetic thermal image"""
        thermal = np.random.rand(*image_size) * 50 + 100
        num_hotspots = np.random.randint(3, 10)
        for _ in range(num_hotspots):
            center_x = np.random.randint(20, image_size[1] - 20)
            center_y = np.random.randint(20, image_size[0] - 20)
            radius = np.random.randint(8, 25)
            hotspot_temp = np.random.uniform(180, 255)

            y, x = np.ogrid[:image_size[0], :image_size[1]]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
            thermal[mask] = hotspot_temp

        thermal_8bit = thermal.astype(np.uint8)
        thermal_colored = cv2.applyColorMap(thermal_8bit, cv2.COLORMAP_INFERNO)
        return cv2.cvtColor(thermal_colored, cv2.COLOR_BGR2RGB)

    def extract_multimodal_features(self, rgb_path=None, thermal_path=None, lidar_path=None):
        """Extract features from all modalities with detailed logging"""
        features = {}
        print(f"Extracting features from RGB:{rgb_path is not None}, Thermal:{thermal_path is not None}, LiDAR:{lidar_path is not None}")

        # RGB features
        if rgb_path and os.path.exists(rgb_path):
            try:
                rgb_image = Image.open(rgb_path).convert('RGB')
                rgb_tensor = self.rgb_transform(rgb_image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    rgb_features = self.rgb_extractor(rgb_tensor)
                    rgb_embedding = self.rgb_siamese.forward_one(rgb_features)

                features['rgb'] = rgb_embedding.cpu().numpy().flatten()
                print("  RGB features extracted successfully")
            except Exception as e:
                print(f"  Error processing RGB: {e}")
                features['rgb'] = np.zeros(512)

        # Thermal features
        if thermal_path and os.path.exists(thermal_path):
            try:
                thermal_image = self.process_thermal_image(thermal_path)
                thermal_tensor = self.thermal_transform(thermal_image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    thermal_features = self.thermal_extractor(thermal_tensor)
                    thermal_embedding = self.thermal_siamese.forward_one(thermal_features)

                features['thermal'] = thermal_embedding.cpu().numpy().flatten()
                print("  Thermal features extracted successfully")
            except Exception as e:
                print(f"  Error processing Thermal: {e}")
                features['thermal'] = np.zeros(512)

        # LiDAR features
        if lidar_path and os.path.exists(lidar_path):
            try:
                lidar_image = self.process_lidar_depth_image(lidar_path)
                lidar_tensor = self.lidar_transform(lidar_image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    lidar_features = self.lidar_extractor(lidar_tensor)
                    lidar_embedding = self.lidar_siamese.forward_one(lidar_features)

                features['lidar'] = lidar_embedding.cpu().numpy().flatten()
                print("  LiDAR features extracted successfully")
            except Exception as e:
                print(f"  Error processing LiDAR: {e}")
                features['lidar'] = np.zeros(512)

        return features

    def visualize_dataset_distribution(self, data_directory, save_path=None):
        """Visualize dataset distribution for paper"""
        print("Creating dataset distribution visualization...")
        
        # Count samples per class
        class_counts = defaultdict(int)
        total_files = 0
        
        for class_dir in os.listdir(data_directory):
            class_path = os.path.join(data_directory, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            # Count PNG files (handle nested structure)
            png_count = 0
            subdirs = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]
            
            if subdirs:
                # Count files in subdirectories
                for subdir in subdirs:
                    subdir_path = os.path.join(class_path, subdir)
                    png_files = [f for f in os.listdir(subdir_path) if f.endswith('.png')]
                    png_count += len(png_files)
            else:
                # Count files directly in class directory
                png_files = [f for f in os.listdir(class_path) if f.endswith('.png')]
                png_count = len(png_files)
            
            class_counts[class_dir] = png_count
            total_files += png_count
        
        # Check if we have valid data
        if not class_counts or sum(class_counts.values()) == 0:
            print("Warning: No valid data found for visualization")
            return None
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        colors = ['#2E8B57', '#FF6B6B', '#4ECDC4']
        
        if counts and max(counts) > 0:
            bars = ax1.bar(classes, counts, color=colors[:len(classes)], alpha=0.8)
            ax1.set_title('Dataset Distribution by Class', fontweight='bold', fontsize=14)
            ax1.set_xlabel('Class', fontweight='bold')
            ax1.set_ylabel('Number of Files', fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(counts),
                        f'{count}', ha='center', va='bottom', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=14)
            ax1.set_title('Dataset Distribution by Class', fontweight='bold', fontsize=14)
        
        # Pie chart
        if counts and sum(counts) > 0:
            # Filter out zero counts
            non_zero_classes = [cls for cls, count in zip(classes, counts) if count > 0]
            non_zero_counts = [count for count in counts if count > 0]
            
            if non_zero_counts:
                wedges, texts, autotexts = ax2.pie(non_zero_counts, labels=non_zero_classes, 
                                                   autopct='%1.1f%%',
                                                   colors=colors[:len(non_zero_classes)], startangle=90)
                ax2.set_title('Dataset Distribution (%)', fontweight='bold', fontsize=14)
            else:
                ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=14)
                ax2.set_title('Dataset Distribution (%)', fontweight='bold', fontsize=14)
        else:
            ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Dataset Distribution (%)', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dataset distribution plot saved to: {save_path}")
        
        plt.show()
        return fig

    def plot_siamese_training_curves(self, training_history, save_path=None):
        """Plot Siamese-specific training curves"""
        print("Creating Siamese training curves visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Siamese Networks Training Progress', fontsize=16, fontweight='bold')
        
        # Extract data
        epochs = range(1, len(training_history['train_losses']) + 1)
        
        # Training Loss
        axes[0, 0].plot(epochs, training_history['train_losses'], 'b-', linewidth=2, label='Training Loss')
        if 'val_losses' in training_history and len(training_history['val_losses']) > 0:
            val_epochs = range(1, len(training_history['val_losses']) + 1)
            if len(val_epochs) == len(epochs):
                axes[0, 0].plot(epochs, training_history['val_losses'], 'r--', linewidth=2, label='Validation Loss')
            else:
                axes[0, 0].plot(val_epochs, training_history['val_losses'], 'r--', linewidth=2, label='Validation Loss')
        axes[0, 0].set_title('Contrastive Loss Curves', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Contrastive Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Modality-specific losses
        if 'modality_losses' in training_history:
            modality_data = training_history['modality_losses']
            for modality, losses in modality_data.items():
                if len(losses) > 0:
                    if len(losses) == len(epochs):
                        axes[0, 1].plot(epochs, losses, linewidth=2, label=f'{modality.upper()} Loss')
                    else:
                        modality_epochs = range(1, len(losses) + 1)
                        axes[0, 1].plot(modality_epochs, losses, linewidth=2, label=f'{modality.upper()} Loss')
            axes[0, 1].set_title('Modality-Specific Losses', fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No modality loss data', ha='center', va='center', 
                           transform=axes[0, 1].transAxes, fontsize=12)
            axes[0, 1].set_title('Modality-Specific Losses', fontweight='bold')
        
        # Learning Rate
        if 'learning_rates' in training_history and len(training_history['learning_rates']) > 0:
            lr_epochs = range(1, len(training_history['learning_rates']) + 1)
            if len(lr_epochs) == len(epochs):
                axes[1, 0].plot(epochs, training_history['learning_rates'], 'purple', linewidth=2)
            else:
                axes[1, 0].plot(lr_epochs, training_history['learning_rates'], 'purple', linewidth=2)
            axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No learning rate data', ha='center', va='center', 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
        
        # Training Info
        info_text = f"Siamese Networks Training\n"
        info_text += f"Total Epochs: {len(epochs)}\n"
        info_text += f"Final Training Loss: {training_history['train_losses'][-1]:.4f}\n"
        if 'modality_losses' in training_history:
            for modality, losses in training_history['modality_losses'].items():
                if len(losses) > 0:
                    info_text += f"Final {modality.upper()} Loss: {losses[-1]:.4f}\n"
        
        axes[1, 1].text(0.5, 0.5, info_text, ha='center', va='center', 
                        transform=axes[1, 1].transAxes, fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1, 1].set_title('Training Summary', fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Siamese training curves plot saved to: {save_path}")
        
        plt.show()
        return fig

    def analyze_dataset_structure(self, data_directory):
        """Analyze and display detailed dataset structure information"""
        print(f"\nAnalyzing dataset structure in: {data_directory}")
        print("="*60)
        
        dataset_info = {
            'classes': {},
            'total_samples': 0,
            'total_files': 0,
            'modality_counts': {'rgb': 0, 'thermal': 0, 'lidar': 0}
        }
        
        for class_dir in os.listdir(data_directory):
            class_path = os.path.join(data_directory, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            print(f"\nClass: {class_dir}")
            print("-" * 40)
            
            # Check for subdirectories
            subdirs = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]
            
            if subdirs:
                print(f"Found {len(subdirs)} sample subdirectories:")
                class_samples = 0
                class_files = 0
                
                for subdir in subdirs:
                    subdir_path = os.path.join(class_path, subdir)
                    files = [f for f in os.listdir(subdir_path) if f.endswith('.png')]
                    
                    rgb_files = [f for f in files if not ('_therm' in f or '_depth' in f)]
                    thermal_files = [f for f in files if '_therm' in f]
                    lidar_files = [f for f in files if '_depth' in f]
                    
                    print(f"  {subdir}: {len(files)} files")
                    print(f"    RGB: {len(rgb_files)}, Thermal: {len(thermal_files)}, LiDAR: {len(lidar_files)}")
                    
                    if len(files) > 0:
                        class_samples += 1
                        class_files += len(files)
                        
                        dataset_info['modality_counts']['rgb'] += len(rgb_files)
                        dataset_info['modality_counts']['thermal'] += len(thermal_files)
                        dataset_info['modality_counts']['lidar'] += len(lidar_files)
                
                dataset_info['classes'][class_dir] = {
                    'samples': class_samples,
                    'files': class_files
                }
                dataset_info['total_samples'] += class_samples
                dataset_info['total_files'] += class_files
                
            else:
                # Flat structure
                files = [f for f in os.listdir(class_path) if f.endswith('.png')]
                rgb_files = [f for f in files if not ('_therm' in f or '_depth' in f)]
                thermal_files = [f for f in files if '_therm' in f]
                lidar_files = [f for f in files if '_depth' in f]
                
                print(f"Flat structure: {len(files)} files")
                print(f"  RGB: {len(rgb_files)}, Thermal: {len(thermal_files)}, LiDAR: {len(lidar_files)}")
                
                dataset_info['classes'][class_dir] = {
                    'samples': len(files) // 3 if len(files) > 0 else 0,  # Estimate
                    'files': len(files)
                }
                dataset_info['total_files'] += len(files)
                dataset_info['modality_counts']['rgb'] += len(rgb_files)
                dataset_info['modality_counts']['thermal'] += len(thermal_files)
                dataset_info['modality_counts']['lidar'] += len(lidar_files)
        
        # Print summary
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"Total classes: {len(dataset_info['classes'])}")
        print(f"Total samples: {dataset_info['total_samples']}")
        print(f"Total files: {dataset_info['total_files']}")
        print(f"\nModality distribution:")
        for modality, count in dataset_info['modality_counts'].items():
            print(f"  {modality.upper()}: {count} files")
        
        return dataset_info

    def plot_training_curves(self, training_history, save_path=None):
        """Plot training curves for paper"""
        print("Creating training curves visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training Progress Analysis', fontsize=16, fontweight='bold')
        
        # Extract data
        epochs = range(1, len(training_history['train_losses']) + 1)
        
        # Training Loss
        axes[0, 0].plot(epochs, training_history['train_losses'], 'b-', linewidth=2, label='Training Loss')
        if 'val_losses' in training_history and len(training_history['val_losses']) > 0:
            # Ensure val_losses has same length as epochs
            val_epochs = range(1, len(training_history['val_losses']) + 1)
            if len(val_epochs) == len(epochs):
                axes[0, 0].plot(epochs, training_history['val_losses'], 'r--', linewidth=2, label='Validation Loss')
            else:
                axes[0, 0].plot(val_epochs, training_history['val_losses'], 'r--', linewidth=2, label='Validation Loss')
        axes[0, 0].set_title('Loss Curves', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy or Pair Accuracy
        if 'val_accuracies' in training_history and len(training_history['val_accuracies']) > 0:
            val_acc_epochs = range(1, len(training_history['val_accuracies']) + 1)
            if len(val_acc_epochs) == len(epochs):
                axes[0, 1].plot(epochs, training_history['val_accuracies'], 'g-', linewidth=2, label='Validation Accuracy')
            else:
                axes[0, 1].plot(val_acc_epochs, training_history['val_accuracies'], 'g-', linewidth=2, label='Validation Accuracy')
            axes[0, 1].set_title('Accuracy Progress', fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        elif 'pair_accuracy' in training_history and len(training_history['pair_accuracy']) > 0:
            pair_acc_epochs = range(1, len(training_history['pair_accuracy']) + 1)
            if len(pair_acc_epochs) == len(epochs):
                axes[0, 1].plot(epochs, training_history['pair_accuracy'], 'g-', linewidth=2, label='Pair Accuracy')
            else:
                axes[0, 1].plot(pair_acc_epochs, training_history['pair_accuracy'], 'g-', linewidth=2, label='Pair Accuracy')
            axes[0, 1].set_title('Siamese Pair Accuracy', fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Pair Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Siamese Networks: Loss-based training\n(No accuracy metrics)', ha='center', va='center', 
                           transform=axes[0, 1].transAxes, fontsize=12)
            axes[0, 1].set_title('Siamese Training Info', fontweight='bold')
        
        # Learning Rate (if available)
        if 'learning_rates' in training_history and len(training_history['learning_rates']) > 0:
            lr_epochs = range(1, len(training_history['learning_rates']) + 1)
            if len(lr_epochs) == len(epochs):
                axes[1, 0].plot(epochs, training_history['learning_rates'], 'purple', linewidth=2)
            else:
                axes[1, 0].plot(lr_epochs, training_history['learning_rates'], 'purple', linewidth=2)
            axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No learning rate data', ha='center', va='center', 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
        
        # Modality-specific losses
        if 'modality_losses' in training_history:
            modality_data = training_history['modality_losses']
            for modality, losses in modality_data.items():
                if len(losses) > 0:
                    if len(losses) == len(epochs):
                        axes[1, 1].plot(epochs, losses, linewidth=2, label=f'{modality.upper()} Loss')
                    else:
                        modality_epochs = range(1, len(losses) + 1)
                        axes[1, 1].plot(modality_epochs, losses, linewidth=2, label=f'{modality.upper()} Loss')
            axes[1, 1].set_title('Modality-Specific Losses', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No modality loss data', ha='center', va='center', 
                           transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('Modality-Specific Losses', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves plot saved to: {save_path}")
        
        plt.show()
        return fig

    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, save_path=None):
        """Plot confusion matrix for paper"""
        print("Creating confusion matrix visualization...")
        
        if class_names is None:
            class_names = ['No Corrosion', 'Mild Corrosion', 'Severe Corrosion']
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Number of Predictions'})
        
        ax.set_title('Confusion Matrix', fontweight='bold', fontsize=14)
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix plot saved to: {save_path}")
        
        plt.show()
        return fig

    def plot_feature_importance(self, feature_names, importance_scores, save_path=None):
        """Plot feature importance for paper"""
        print("Creating feature importance visualization...")
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        })
        df = df.sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create horizontal bar plot
        bars = ax.barh(df['Feature'], df['Importance'], color='skyblue', alpha=0.8)
        
        ax.set_title('Feature Importance Analysis', fontweight='bold', fontsize=14)
        ax.set_xlabel('Importance Score', fontweight='bold')
        ax.set_ylabel('Features', fontweight='bold')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")
        
        plt.show()
        return fig

    def plot_accuracy_comparison(self, results_dict, save_path=None):
        """Plot accuracy comparison between different approaches"""
        print("Creating accuracy comparison visualization...")
        
        methods = list(results_dict.keys())
        accuracies = list(results_dict.values())
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        colors = ['#2E8B57', '#FF6B6B', '#4ECDC4', '#FFD93D', '#6C5CE7']
        bars = ax.bar(methods, accuracies, color=colors[:len(methods)], alpha=0.8)
        
        ax.set_title('Accuracy Comparison Across Methods', fontweight='bold', fontsize=14)
        ax.set_xlabel('Method', fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Accuracy comparison plot saved to: {save_path}")
        
        plt.show()
        return fig

    def create_siamese_pairs(self, dataset, positive_ratio=0.5):
        """Create positive and negative pairs for Siamese training"""
        print(f"\nCreating Siamese pairs from {len(dataset)} samples...")

        pairs = []
        labels = []

        # Group samples by class
        class_samples = defaultdict(list)
        for sample in dataset:
            class_samples[sample['class']].append(sample)

        print("Samples per class:")
        for class_label, samples in class_samples.items():
            print(f"  Class {class_label}: {len(samples)} samples")

        num_pairs = len(dataset) * 2  # Create more pairs
        num_positive = int(num_pairs * positive_ratio)
        num_negative = num_pairs - num_positive

        print(f"Creating {num_positive} positive pairs and {num_negative} negative pairs...")

        # Create positive pairs (same class)
        positive_created = 0
        for _ in range(num_positive):
            class_label = random.choice(list(class_samples.keys()))
            samples = class_samples[class_label]

            if len(samples) >= 2:
                sample1, sample2 = random.sample(samples, 2)
                pairs.append((sample1, sample2))
                labels.append(1)
                positive_created += 1

        # Create negative pairs (different classes)
        negative_created = 0
        for _ in range(num_negative):
            if len(class_samples) >= 2:
                class1, class2 = random.sample(list(class_samples.keys()), 2)
                sample1 = random.choice(class_samples[class1])
                sample2 = random.choice(class_samples[class2])
                pairs.append((sample1, sample2))
                labels.append(0)
                negative_created += 1

        print(f"Created {positive_created} positive pairs and {negative_created} negative pairs")
        return pairs, labels

    def train_siamese_networks(self, data_directory, epochs=100, lr=0.001, batch_size=16):
        """Complete training pipeline with extensive logging"""
        print("="*50)
        print("STARTING SIAMESE NETWORK TRAINING")
        print("="*50)

        # Check dataset
        if not self.check_dataset_structure(data_directory):
            print("Dataset check failed!")
            return None

        # Prepare dataset
        train_data, val_data = self.prepare_training_dataset(data_directory)

        if len(train_data) == 0:
            print("ERROR: No training data found!")
            return None

        # Create training pairs
        train_pairs, train_labels = self.create_siamese_pairs(train_data)
        val_pairs, val_labels = self.create_siamese_pairs(val_data) if val_data else ([], [])

        if len(train_pairs) == 0:
            print("ERROR: No training pairs created!")
            return None

        print(f"Training with {len(train_pairs)} pairs, validating with {len(val_pairs)} pairs")

        # Initialize optimizers
        optimizers = {
            'rgb': torch.optim.Adam(self.rgb_siamese.parameters(), lr=lr),
            'thermal': torch.optim.Adam(self.thermal_siamese.parameters(), lr=lr),
            'lidar': torch.optim.Adam(self.lidar_siamese.parameters(), lr=lr)
        }

        criterion = ContrastiveLoss(margin=2.0)

        # Training loop
        print(f"\nStarting training for {epochs} epochs...")
        
        # Initialize training history
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'modality_losses': {'rgb': [], 'thermal': [], 'lidar': []},
            'learning_rates': [],
            'pair_accuracy': []  # Add pair accuracy tracking
        }

        for epoch in range(epochs):
            total_loss = {'rgb': 0, 'thermal': 0, 'lidar': 0}
            processed_pairs = 0
            correct_pairs = 0

            # Process in smaller batches to show progress
            for i in range(0, len(train_pairs), batch_size):
                batch_pairs = train_pairs[i:i+batch_size]
                batch_labels = train_labels[i:i+batch_size]

                for pair, label in zip(batch_pairs, batch_labels):
                    try:
                        # Extract features
                        features1 = self.extract_multimodal_features(
                            rgb_path=pair[0].get('rgb'),
                            thermal_path=pair[0].get('thermal'),
                            lidar_path=pair[0].get('lidar')
                        )
                        features2 = self.extract_multimodal_features(
                            rgb_path=pair[1].get('rgb'),
                            thermal_path=pair[1].get('thermal'),
                            lidar_path=pair[1].get('lidar')
                        )

                        # Train each modality
                        for modality in ['rgb', 'thermal', 'lidar']:
                            if modality in features1 and modality in features2:
                                optimizer = optimizers[modality]
                                siamese_net = getattr(self, f'{modality}_siamese')

                                feat1 = torch.FloatTensor(features1[modality]).unsqueeze(0).to(self.device)
                                feat2 = torch.FloatTensor(features2[modality]).unsqueeze(0).to(self.device)
                                label_tensor = torch.FloatTensor([label]).to(self.device)

                                optimizer.zero_grad()
                                emb1, emb2 = siamese_net(feat1, feat2)
                                loss = criterion(emb1, emb2, label_tensor)
                                loss.backward()
                                optimizer.step()

                                total_loss[modality] += loss.item()

                        processed_pairs += 1

                    except Exception as e:
                        print(f"Error processing pair {processed_pairs}: {e}")
                        continue

            # Calculate average losses and store in history
            if processed_pairs > 0:
                avg_total_loss = sum(total_loss.values()) / len(total_loss)
                training_history['train_losses'].append(avg_total_loss)
                
                for modality, loss_val in total_loss.items():
                    avg_loss = loss_val / processed_pairs
                    training_history['modality_losses'][modality].append(avg_loss)
                
                # Store learning rate
                training_history['learning_rates'].append(lr)

            # Print progress
            if epoch % 10 == 0 or epoch < 5:
                print(f"\nEpoch {epoch}/{epochs}:")
                for modality, loss_val in total_loss.items():
                    if processed_pairs > 0:
                        avg_loss = loss_val / processed_pairs
                        print(f"  {modality} avg loss: {avg_loss:.4f}")
                    else:
                        print(f"  {modality}: No pairs processed")
                print(f"  Processed {processed_pairs} pairs")

                # Force output in Colab
                if IN_COLAB:
                    import sys
                    sys.stdout.flush()

        print("Siamese network training completed!")
        return training_history

    def save_models(self, save_directory):
        """Save trained models"""
        try:
            os.makedirs(save_directory, exist_ok=True)

            torch.save(self.rgb_siamese.state_dict(),
                      os.path.join(save_directory, 'rgb_siamese.pth'))
            torch.save(self.thermal_siamese.state_dict(),
                      os.path.join(save_directory, 'thermal_siamese.pth'))
            torch.save(self.lidar_siamese.state_dict(),
                      os.path.join(save_directory, 'lidar_siamese.pth'))
            torch.save(self.fusion_net.state_dict(),
                      os.path.join(save_directory, 'fusion_net.pth'))

            print(f"✓ Models saved to {save_directory}")

        except Exception as e:
            print(f"✗ Error saving models: {e}")

    def load_models(self, save_directory):
        """Load pre-trained models"""
        try:
            rgb_path = os.path.join(save_directory, 'rgb_siamese.pth')
            thermal_path = os.path.join(save_directory, 'thermal_siamese.pth')
            lidar_path = os.path.join(save_directory, 'lidar_siamese.pth')
            fusion_path = os.path.join(save_directory, 'fusion_net.pth')

            if os.path.exists(rgb_path):
                self.rgb_siamese.load_state_dict(torch.load(rgb_path, map_location=self.device))
                print("✓ RGB Siamese model loaded")

            if os.path.exists(thermal_path):
                self.thermal_siamese.load_state_dict(torch.load(thermal_path, map_location=self.device))
                print("✓ Thermal Siamese model loaded")

            if os.path.exists(lidar_path):
                self.lidar_siamese.load_state_dict(torch.load(lidar_path, map_location=self.device))
                print("✓ LiDAR Siamese model loaded")

            if os.path.exists(fusion_path):
                self.fusion_net.load_state_dict(torch.load(fusion_path, map_location=self.device))
                print("✓ Fusion network loaded")

        except Exception as e:
            print(f"✗ Error loading models: {e}")

    def fuse_multimodal_features(self, features_dict):
        """Fuse multi-modal features using fusion network"""
        try:
            # Prepare input for fusion network
            modality_embeddings = []

            for modality in ['rgb', 'thermal', 'lidar']:
                if modality in features_dict:
                    modality_embeddings.append(features_dict[modality])
                else:
                    modality_embeddings.append(np.zeros(512))  # Zero padding for missing modalities

            # Concatenate embeddings
            fused_input = np.concatenate(modality_embeddings)
            fused_tensor = torch.FloatTensor(fused_input).unsqueeze(0).to(self.device)

            # Forward through fusion network
            with torch.no_grad():
                fused_embedding = self.fusion_net(fused_tensor)

            return fused_embedding.cpu().numpy().flatten()

        except Exception as e:
            print(f"Error in feature fusion: {e}")
            return np.zeros(128)  # Return zero embedding as fallback

    def train_fusion_network(self, data_directory, epochs=100, lr=0.001, batch_size=32):
        """
        Stage 2: Train fusion network for corrosion classification
        Prerequisite: Siamese networks must be trained first
        """
        print("="*50)
        print("STARTING FUSION NETWORK TRAINING (STAGE 2)")
        print("="*50)
        
        # Prepare dataset
        train_data, val_data = self.prepare_training_dataset(data_directory)
        
        if len(train_data) == 0:
            print("ERROR: No training data found!")
            return None
        
        print(f"Training fusion network with {len(train_data)} samples")
        
        # Add classification head to fusion network
        self.fusion_classifier = nn.Sequential(
            self.fusion_net,
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 3)  # 3 classes: no_corrosion, mild, severe
        ).to(self.device)
        
        # Setup optimizer and loss
        fusion_optimizer = torch.optim.Adam(self.fusion_classifier.parameters(), lr=lr)
        fusion_criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_accuracy = 0.0
        train_losses = []
        val_accuracies = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            self.fusion_classifier.train()
            epoch_loss = 0.0
            processed_samples = 0
            
            # Shuffle training data
            random.shuffle(train_data)
            
            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i:i+batch_size]
                
                batch_features = []
                batch_labels = []
                
                for sample in batch_data:
                    try:
                        # Extract multi-modal features using trained Siamese networks
                        features = self.extract_multimodal_features(
                            rgb_path=sample.get('rgb'),
                            thermal_path=sample.get('thermal'),
                            lidar_path=sample.get('lidar')
                        )
                        
                        # Prepare features for fusion
                        feature_vector = []
                        for modality in ['rgb', 'thermal', 'lidar']:
                            if modality in features:
                                feature_vector.extend(features[modality])
                            else:
                                feature_vector.extend(np.zeros(512))
                        
                        batch_features.append(feature_vector)
                        batch_labels.append(sample['class'])
                        processed_samples += 1
                        
                    except Exception as e:
                        print(f"Error processing sample: {e}")
                        continue
                
                if len(batch_features) > 0:
                    # Convert to tensors
                    features_tensor = torch.FloatTensor(batch_features).to(self.device)
                    labels_tensor = torch.LongTensor(batch_labels).to(self.device)
                    
                    # Forward pass
                    fusion_optimizer.zero_grad()
                    outputs = self.fusion_classifier(features_tensor)
                    loss = fusion_criterion(outputs, labels_tensor)
                    
                    # Backward pass
                    loss.backward()
                    fusion_optimizer.step()
                    
                    epoch_loss += loss.item()
            
            # Validation phase
            val_accuracy = 0.0
            val_loss = 0.0
            if val_data:
                val_accuracy, val_loss = self.evaluate_fusion_network(val_data)
                val_accuracies.append(val_accuracy)
                val_losses.append(val_loss)
                
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    # Save best model
                    torch.save(self.fusion_classifier.state_dict(), 
                            os.path.join('/content', 'best_fusion_classifier.pth'))
            
            avg_loss = epoch_loss / max(1, len(train_data) // batch_size)
            train_losses.append(avg_loss)
            
            # Print progress
            if epoch % 10 == 0 or epoch < 5:
                print(f"Epoch {epoch}/{epochs}:")
                print(f"  Training Loss: {avg_loss:.4f}")
                print(f"  Validation Accuracy: {val_accuracy:.4f}")
                print(f"  Validation Loss: {val_loss:.4f}")
                print(f"  Processed Samples: {processed_samples}")
                print(f"  Best Accuracy: {best_accuracy:.4f}")
        
        print(f"\nFusion network training completed!")
        print(f"Best validation accuracy: {best_accuracy:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_accuracy': best_accuracy
        }

    def evaluate_fusion_network(self, val_data):
        """Evaluate fusion network on validation data"""
        self.fusion_classifier.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for sample in val_data:
                try:
                    # Extract features
                    features = self.extract_multimodal_features(
                        rgb_path=sample.get('rgb'),
                        thermal_path=sample.get('thermal'),
                        lidar_path=sample.get('lidar')
                    )
                    
                    # Prepare feature vector
                    feature_vector = []
                    for modality in ['rgb', 'thermal', 'lidar']:
                        if modality in features:
                            feature_vector.extend(features[modality])
                        else:
                            feature_vector.extend(np.zeros(512))
                    
                    # Predict
                    features_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(self.device)
                    outputs = self.fusion_classifier(features_tensor)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    # Calculate loss
                    label_tensor = torch.LongTensor([sample['class']]).to(self.device)
                    loss = criterion(outputs, label_tensor)
                    total_loss += loss.item()
                    
                    total += 1
                    if predicted.item() == sample['class']:
                        correct += 1
                        
                except Exception as e:
                    continue
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        return accuracy, avg_loss

    def complete_training_pipeline(self, data_directory, 
                                siamese_epochs=100, fusion_epochs=100,
                                siamese_lr=0.001, fusion_lr=0.001,
                                save_plots=True):
        """
        Complete two-stage training pipeline with visualization
        """
        print("STARTING COMPLETE TRAINING PIPELINE")
        print("="*60)
        
        # Visualize dataset distribution
        print("Creating dataset visualization...")
        self.visualize_dataset_distribution(
            data_directory, 
            save_path='/content/dataset_distribution.png' if save_plots else None
        )
        
        # Stage 1: Train Siamese Networks
        print("STAGE 1: Training Siamese Networks...")
        siamese_results = self.train_siamese_networks(
            data_directory, 
            epochs=siamese_epochs, 
            lr=siamese_lr
        )
        
        if siamese_results is None:
            print("ERROR: Siamese training failed!")
            return None
        
        print("✓ Siamese networks training completed!")
        
        # Plot Siamese training curves
        if save_plots:
            self.plot_training_curves(
                siamese_results,
                save_path='/content/siamese_training_curves.png'
            )
        
        # Save Siamese models
        self.save_models('/content/siamese_models')
        
        # Stage 2: Train Fusion Network
        print("\nSTAGE 2: Training Fusion Network...")
        fusion_results = self.train_fusion_network(
            data_directory,
            epochs=fusion_epochs,
            lr=fusion_lr
        )
        
        if fusion_results is None:
            print("ERROR: Fusion training failed!")
            return None
        
        print("✓ Fusion network training completed!")
        
        # Plot Fusion training curves
        if save_plots and fusion_results:
            self.plot_training_curves(
                fusion_results,
                save_path='/content/fusion_training_curves.png'
            )
        
        return {
            'siamese_results': siamese_results,
            'fusion_results': fusion_results
        }

    def predict_corrosion(self, rgb_path=None, thermal_path=None, lidar_path=None):
        """
        Complete prediction pipeline using both trained networks
        """
        try:
            print("Making corrosion prediction...")
            
            # Stage 1: Extract features using trained Siamese networks
            print("Extracting multi-modal features...")
            features = self.extract_multimodal_features(rgb_path, thermal_path, lidar_path)
            
            # Stage 2: Prepare features for fusion network
            feature_vector = []
            for modality in ['rgb', 'thermal', 'lidar']:
                if modality in features:
                    feature_vector.extend(features[modality])
                    print(f"✓ {modality} features: {len(features[modality])} dims")
                else:
                    feature_vector.extend(np.zeros(512))
                    print(f"✗ {modality} features: using zeros")
            
            print(f"Total fused feature vector: {len(feature_vector)} dims")
            
            # Stage 3: Classify using fusion network
            self.fusion_classifier.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(self.device)
                outputs = self.fusion_classifier(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
            
            # Convert to interpretable results
            class_names = ['No Corrosion', 'Mild Corrosion', 'Severe Corrosion']
            predicted_class = class_names[predicted.item()]
            confidence_scores = probabilities.cpu().numpy().flatten()
            
            results = {
                'predicted_class': predicted_class,
                'predicted_index': predicted.item(),
                'confidence_scores': {
                    'No Corrosion': confidence_scores[0],
                    'Mild Corrosion': confidence_scores[1],
                    'Severe Corrosion': confidence_scores[2]
                },
                'max_confidence': confidence_scores.max()
            }
            
            print(f"Prediction: {predicted_class}")
            print(f"Confidence: {results['max_confidence']:.4f}")
            
            return results
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None

    def save_complete_model(self, save_directory):
        """Save all trained components"""
        try:
            os.makedirs(save_directory, exist_ok=True)
            
            # Save Siamese networks
            self.save_models(save_directory)
            
            # Save fusion classifier
            if hasattr(self, 'fusion_classifier'):
                torch.save(self.fusion_classifier.state_dict(),
                        os.path.join(save_directory, 'fusion_classifier.pth'))
                print("✓ Fusion classifier saved")
            
            print(f"✓ Complete model saved to {save_directory}")
            
        except Exception as e:
            print(f"✗ Error saving complete model: {e}")

    def load_complete_model(self, save_directory):
        """Load all trained components"""
        try:
            # Load Siamese networks
            self.load_models(save_directory)
            
            # Load fusion classifier
            fusion_path = os.path.join(save_directory, 'fusion_classifier.pth')
            if os.path.exists(fusion_path):
                # Recreate fusion classifier architecture
                self.fusion_classifier = nn.Sequential(
                    self.fusion_net,
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(64, 3)
                ).to(self.device)
                
                self.fusion_classifier.load_state_dict(torch.load(fusion_path, map_location=self.device))
                print("✓ Fusion classifier loaded")
            
            print("✓ Complete model loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading complete model: {e}")
# Training pipeline with error handling and visualization
def train_complete_system():
    """Complete training pipeline with comprehensive error handling and visualization"""
    try:
        print("="*60)
        print("INITIALIZING MULTIMODAL CORROSION ANALYZER")
        print("="*60)

        # Initialize analyzer
        analyzer = MultiModalCorrosionAnalyzer()

        # Set data directory
        data_directory = "/content/drive/MyDrive/S3 UTP/MS2_dataset/dataset"

        print(f"\nData directory set to: {data_directory}")

        # Check if directory exists
        if not os.path.exists(data_directory):
            print(f"ERROR: Data directory {data_directory} does not exist!")
            return None, None

        # Check dataset structure
        if not analyzer.check_dataset_structure(data_directory):
            print("ERROR: No valid dataset found!")
            return None, None

        print("\n" + "="*60)
        print("STARTING COMPLETE TRAINING PIPELINE")
        print("="*60)

        # Run complete training pipeline with visualization
        results = analyzer.complete_training_pipeline(
            data_directory=data_directory,
            siamese_epochs=50,  # Reduced for testing
            fusion_epochs=50,    # Reduced for testing
            siamese_lr=0.001,
            fusion_lr=0.001,
            save_plots=True
        )

        if results is None:
            print("Training pipeline failed!")
            return None, None

        print("Complete training pipeline finished successfully!")

        # Save models
        print("Saving trained models...")
        model_dir = "/content/trained_models"
        os.makedirs(model_dir, exist_ok=True)
        analyzer.save_complete_model(model_dir)

        return analyzer, results

    except Exception as e:
        print(f"ERROR in training pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def safe_training():
    """Safe training dengan error handling yang lebih baik"""
    try:
        analyzer = MultiModalCorrosionAnalyzer()
        data_directory = "/content/drive/MyDrive/S3 UTP/MS2_dataset/dataset"
        
        # Prepare dataset
        train_data, val_data = analyzer.prepare_training_dataset(data_directory)
        print(f"Dataset prepared: {len(train_data)} train, {len(val_data)} val")
        
        if len(train_data) == 0:
            print("ERROR: No training data!")
            return None, None
        
        # Train Siamese networks
        print("Starting Siamese training...")
        results = analyzer.train_siamese_networks(
            data_directory=data_directory,
            epochs=20,  # Reduced
            lr=0.001,
            batch_size=4
        )
        
        if results:
            print("Training completed successfully!")
            print("Results keys:", list(results.keys()))
            
            # Plot results
            try:
                # Use Siamese-specific visualization
                analyzer.plot_siamese_training_curves(results)
                print("Siamese training visualization completed!")
            except Exception as e:
                print(f"Visualization error: {e}")
                # Fallback to general visualization
                try:
                    analyzer.plot_training_curves(results)
                    print("Fallback visualization completed!")
                except Exception as e2:
                    print(f"Fallback visualization also failed: {e2}")
        
        return analyzer, results
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Jalankan safe training
analyzer, results = safe_training()






def train_siamese_only():
    """Train only Siamese networks without fusion"""
    try:
        print("="*60)
        print("TRAINING SIAMESE NETWORKS ONLY")
        print("="*60)

        # Initialize analyzer
        analyzer = MultiModalCorrosionAnalyzer()

        # Set data directory
        data_directory = "/content/drive/MyDrive/S3 UTP/MS2_dataset/dataset"

        print(f"\nData directory set to: {data_directory}")

        # Check if directory exists
        if not os.path.exists(data_directory):
            print(f"ERROR: Data directory {data_directory} does not exist!")
            return None, None

        # Check dataset structure
        if not analyzer.check_dataset_structure(data_directory):
            print("ERROR: No valid dataset found!")
            return None, None

        print("\n" + "="*60)
        print("STARTING SIAMESE TRAINING")
        print("="*60)

        # Train Siamese networks only
        siamese_results = analyzer.train_siamese_networks(
            data_directory=data_directory,
            epochs=50,
            lr=0.001,
            batch_size=4
        )

        if siamese_results is None:
            print("Siamese training failed!")
            return None, None

        print("Siamese training completed successfully!")

        # Plot training curves
        analyzer.plot_training_curves(
            siamese_results,
            save_path='/content/siamese_training_curves.png'
        )

        # Save models
        print("Saving trained models...")
        model_dir = "/content/siamese_models"
        os.makedirs(model_dir, exist_ok=True)
        analyzer.save_models(model_dir)

        return analyzer, siamese_results

    except Exception as e:
        print(f"ERROR in Siamese training: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Quick test function
def quick_test():
    """Quick test to verify everything works"""
    print("Running quick test...")

    try:
        # Test basic initialization
        analyzer = MultiModalCorrosionAnalyzer()
        print("✓ Analyzer initialized")

        # Test dataset structure check
        data_directory = "/content/drive/MyDrive/S3 UTP/MS2_dataset/dataset"
        if os.path.exists(data_directory):
            # Analyze dataset structure
            dataset_info = analyzer.analyze_dataset_structure(data_directory)
            print("✓ Dataset analysis completed")
            
            # Test dataset preparation
            train_data, val_data = analyzer.prepare_training_dataset(data_directory)
            print(f"✓ Dataset preparation completed: {len(train_data)} train, {len(val_data)} val")
            
            # Test visualization
            analyzer.visualize_dataset_distribution(data_directory)
            print("✓ Dataset visualization completed")
            
        else:
            print("⚠ Dataset directory not found, skipping dataset test")

        # Test visualization methods
        print("Testing visualization methods...")
        # Create sample data for visualization test
        sample_history = {
            'train_losses': [0.5, 0.4, 0.3, 0.25, 0.2],
            'val_losses': [0.6, 0.5, 0.4, 0.35, 0.3],
            'val_accuracies': [0.6, 0.7, 0.8, 0.85, 0.9],
            'modality_losses': {
                'rgb': [0.5, 0.4, 0.3, 0.25, 0.2],
                'thermal': [0.6, 0.5, 0.4, 0.35, 0.3],
                'lidar': [0.7, 0.6, 0.5, 0.45, 0.4]
            }
        }
        
        # Test training curves plot
        analyzer.plot_training_curves(sample_history)
        print("✓ Training curves visualization test completed")

        print("All tests passed! Ready for training.")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def evaluate_and_visualize_results(analyzer, test_data_directory, save_plots=True):
    """
    Comprehensive evaluation and visualization for paper
    """
    print("="*60)
    print("COMPREHENSIVE EVALUATION AND VISUALIZATION")
    print("="*60)
    
    try:
        # Load test data
        test_data, _ = analyzer.prepare_training_dataset(test_data_directory, validation_split=0.0)
        
        if len(test_data) == 0:
            print("ERROR: No test data found!")
            return None
        
        print(f"Evaluating on {len(test_data)} test samples...")
        
        # Collect predictions and true labels
        y_true = []
        y_pred = []
        confidence_scores = []
        
        for sample in test_data:
            try:
                # Make prediction
                result = analyzer.predict_corrosion(
                    rgb_path=sample.get('rgb'),
                    thermal_path=sample.get('thermal'),
                    lidar_path=sample.get('lidar')
                )
                
                if result:
                    y_true.append(sample['class'])
                    y_pred.append(result['predicted_index'])
                    confidence_scores.append(result['max_confidence'])
                    
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
        
        if len(y_true) == 0:
            print("ERROR: No predictions made!")
            return None
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        print(f"\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Create visualizations
        if save_plots:
            # Confusion Matrix
            analyzer.plot_confusion_matrix(
                y_true, y_pred,
                save_path='/content/confusion_matrix.png'
            )
            
            # Confidence Distribution
            plt.figure(figsize=(10, 6))
            plt.hist(confidence_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Prediction Confidence Distribution', fontweight='bold', fontsize=14)
            plt.xlabel('Confidence Score', fontweight='bold')
            plt.ylabel('Frequency', fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.savefig('/content/confidence_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Accuracy by Class
            class_names = ['No Corrosion', 'Mild Corrosion', 'Severe Corrosion']
            class_accuracies = []
            
            for i in range(3):
                class_mask = [y == i for y in y_true]
                if any(class_mask):
                    class_pred = [y_pred[j] for j, mask in enumerate(class_mask) if mask]
                    class_true = [y_true[j] for j, mask in enumerate(class_mask) if mask]
                    class_acc = sum(1 for p, t in zip(class_pred, class_true) if p == t) / len(class_true)
                    class_accuracies.append(class_acc)
                else:
                    class_accuracies.append(0.0)
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(class_names, class_accuracies, color=['#2E8B57', '#FF6B6B', '#4ECDC4'], alpha=0.8)
            plt.title('Accuracy by Corrosion Class', fontweight='bold', fontsize=14)
            plt.xlabel('Corrosion Class', fontweight='bold')
            plt.ylabel('Accuracy', fontweight='bold')
            plt.ylim(0, 1)
            
            # Add value labels
            for bar, acc in zip(bars, class_accuracies):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('/content/accuracy_by_class.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_true': y_true,
            'y_pred': y_pred,
            'confidence_scores': confidence_scores
        }
        
    except Exception as e:
        print(f"ERROR in evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None






