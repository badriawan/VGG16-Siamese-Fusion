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

# Import custom classes from separated files
from VGG16 import VGG16FeatureExtractor
from siamese import SiameseNetwork
from loss import ContrastiveLoss
from analyzer import MultiModalCorrosionAnalyzer

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

def test_imports():
    """Test if all imports are working correctly"""
    try:
        print("Testing imports...")
        
        # Test PyTorch and related imports
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ Device: {device}")
        
        # Test custom class imports
        vgg_extractor = VGG16FeatureExtractor()
        print("✓ VGG16FeatureExtractor imported and initialized")
        
        siamese_net = SiameseNetwork()
        print("✓ SiameseNetwork imported and initialized")
        
        contrastive_loss = ContrastiveLoss()
        print("✓ ContrastiveLoss imported and initialized")
        
        analyzer = MultiModalCorrosionAnalyzer()
        print("✓ MultiModalCorrosionAnalyzer imported and initialized")
        
        print("All imports successful! ✅")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_dataset():
    """Visualize dataset"""
    try:
        print("="*60)
        print("VISUALIZE DATASET")
        print("="*60)
        
        # Initialize analyzer
        analyzer = MultiModalCorrosionAnalyzer()
        
        # Data directory for temporal dataset structure
        data_directory = "/content/drive/MyDrive/Colab Notebooks/temporal_dataset"
        
        print(f"\nExpected dataset structure:")
        print("temporal_dataset/")
        print("├── sample_001/")
        print("│   ├── before/")
        print("│   │   ├── rgb.png")
        print("│   │   ├── thermal.png")
        print("│   │   └── depth.png (LIDAR)")
        print("│   ├── after/")
        print("│   │   ├── rgb.png")
        print("│   │   ├── thermal.png")
        print("│   │   └── depth.png (LIDAR)")
        print("│   └── label.txt      # 0 = no change, 1 = change detected")
        print(f"\nData directory: {data_directory}")
        
        # Check if dataset exists
        if not os.path.exists(data_directory):
            print(f"\nWARNING: Dataset not found at {data_directory}")
            print("Please prepare your dataset according to the structure above.")
            return None, None
        
        # Visualize temporal dataset
        results = analyzer.visualize_temporal_dataset_distribution(
            data_directory=data_directory,
        )
        
        if results:
            print("\n✓ Visualize dataset completed successfully!")
            print(f"Visualization result type: {type(results)}")
            if hasattr(results, 'keys'):
                print("Results keys:", list(results.keys()))
            else:
                print("Visualization plot created successfully!")

        
        return analyzer, results
        
    except Exception as e:
        print(f"Visualize dataset error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def train_change_detection():
    """Train change detection system"""
    try:
        print("="*60)
        print("CHANGE DETECTION TRAINING")
        print("Objective: Detect corrosion growth between before/after timepoints")
        print("="*60)
        
        # Initialize analyzer
        analyzer = MultiModalCorrosionAnalyzer()
        
        # Data directory for temporal dataset structure
        data_directory = "/content/drive/MyDrive/Colab Notebooks/temporal_dataset"
        
        print(f"\nExpected dataset structure:")
        print("temporal_dataset/")
        print("├── sample_001/")
        print("│   ├── before/")
        print("│   │   ├── rgb.png")
        print("│   │   ├── thermal.png")
        print("│   │   └── depth.png (LIDAR)")
        print("│   ├── after/")
        print("│   │   ├── rgb.png")
        print("│   │   ├── thermal.png")
        print("│   │   └── depth.png (LIDAR)")
        print("│   └── label.txt      # 0 = no change, 1 = change detected")
        print(f"\nData directory: {data_directory}")
        
        # Check if dataset exists
        if not os.path.exists(data_directory):
            print(f"\nWARNING: Dataset not found at {data_directory}")
            print("Please prepare your dataset according to the structure above.")
            return None, None
        
        # Train complete system
        results = analyzer.train_change_detection(
            data_directory=data_directory,
            epochs=50,  # Reduced for testing
            lr=0.001,
            batch_size=8
        )
        
        if results:
            print("\n✓ Change detection training completed successfully!")
            print("Results keys:", list(results.keys()))
            
            # Save trained model
            model_dir = "/content/trained_models"
            os.makedirs(model_dir, exist_ok=True)
            analyzer.save_temporal_model(model_dir)
            print(f"✓ Model saved to: {model_dir}")
        
        return analyzer, results
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_system():
    """Test system functionality"""
    print("Running system test...")
    
    try:
        # Test basic initialization
        analyzer = MultiModalCorrosionAnalyzer()
        print("✓ Analyzer initialized")
        
        # Test methods
        methods_to_test = [
            'prepare_training_dataset',
            'extract_multimodal_features', 
            'train_change_detection',
            'predict_temporal_change'
        ]
        
        for method in methods_to_test:
            if hasattr(analyzer, method):
                print(f"✓ Method {method} exists")
            else:
                print(f"✗ Method {method} missing")
        
        # Test dataset structure check
        data_directory = "/content/drive/MyDrive/Colab Notebooks/temporal_dataset"
        if os.path.exists(data_directory):
            print("✓ Dataset directory found")
            
            # Test dataset preparation
            train_data, val_data = analyzer.prepare_training_dataset(data_directory)
            print(f"✓ Dataset preparation: {len(train_data)} train, {len(val_data)} val")
            
        else:
            print("⚠ Dataset directory not found, skipping dataset test")
        
        print("System test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def predict_change(analyzer, before_paths, after_paths):
    """Make change prediction on new data"""
    try:
        print("Making change prediction...")
        
        if not analyzer:
            print("ERROR: No trained analyzer provided!")
            return None
        
        # Make prediction
        prediction = analyzer.predict_temporal_change(
            before_paths=before_paths,
            after_paths=after_paths
        )
        
        if prediction:
            print(f"\n--- PREDICTION RESULT ---")
            print(f"Change Status: {prediction['predicted_class']}")
            print(f"Confidence: {prediction['max_confidence']:.4f}")
            print(f"Confidence Scores:")
            for class_name, score in prediction['confidence_scores'].items():
                print(f"  {class_name}: {score:.4f}")
        
        return prediction
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def evaluate_change_detection(analyzer, test_data_directory, save_plots=True):
    """Evaluate change detection system"""
    print("="*60)
    print("CHANGE DETECTION EVALUATION")
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
                result = analyzer.predict_temporal_change(
                    before_paths=sample['before'],
                    after_paths=sample['after']
                )
                
                if result:
                    y_true.append(sample['change_label'])
                    y_pred.append(result['predicted_index'])
                    confidence_scores.append(result['max_confidence'])
                    
            except Exception as e:
                print(f"Error processing sample {sample['sample_id']}: {e}")
                continue
        
        if len(y_true) == 0:
            print("ERROR: No predictions made!")
            return None
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        print(f"\nChange Detection Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Detailed classification report
        class_names = ['No Change', 'Change Detected']
        report = classification_report(y_true, y_pred, target_names=class_names)
        print(f"\nDetailed Classification Report:")
        print(report)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_true': y_true,
            'y_pred': y_pred,
            'confidence_scores': confidence_scores,
            'classification_report': report
        }
        
    except Exception as e:
        print(f"ERROR in evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function for change detection system"""
    print("="*60)
    print("CORROSION CHANGE DETECTION ANALYZER")
    print("Objective: Detect corrosion growth progression")
    print("Input: Before/After temporal pairs (RGB, Thermal, LIDAR)")
    print("Output: Binary classification (Change/No Change)")
    print("="*60)
    
    # Test imports first
    if not test_imports():
        print("❌ Import test failed! Please check your file structure.")
        return
    
    print("\nAvailable Options:")
    print("1. Test system functionality")
    print("2. Train change detection system")
    print("3. Evaluate trained system")
    print("4. Make prediction on new data")
    
    # For demonstration, run system test
    print("\nRunning system test...")
    test_system()

# Usage examples for change detection
if __name__ == "__main__":
    # Test imports
    test_imports()
    
    # Available usage patterns:
    
    # Example 1: Test system
    # test_system()
    
    # Example 2: Train change detection
    # analyzer, results = train_change_detection()
    
    # Example 3: Evaluate system
    # if analyzer:
    #     test_dir = "/content/drive/MyDrive/S3 UTP/temporal_test_dataset"
    #     evaluation_results = evaluate_change_detection(analyzer, test_dir)
    
    # Example 4: Make predictions
    # if analyzer:
    #     before_paths = {
    #         'rgb': '/path/to/before_rgb.png',
    #         'thermal': '/path/to/before_thermal.png',
    #         'lidar': '/path/to/before_depth.png'
    #     }
    #     after_paths = {
    #         'rgb': '/path/to/after_rgb.png',
    #         'thermal': '/path/to/after_thermal.png', 
    #         'lidar': '/path/to/after_depth.png'
    #     }
    #     prediction = predict_change(analyzer, before_paths, after_paths)

# Manual usage examples for change detection:
"""
CHANGE DETECTION WORKFLOW:

# 1. Test system
test_imports()
test_system()

# 2. Train change detection system
analyzer, results = train_change_detection()

# 3. Evaluate system
if analyzer:
    test_dir = "/content/drive/MyDrive/S3 UTP/temporal_test_dataset"
    evaluation_results = evaluate_change_detection(analyzer, test_dir)

# 4. Make predictions on new temporal pairs
if analyzer:
    before_paths = {
        'rgb': '/path/to/before_rgb.png',
        'thermal': '/path/to/before_thermal.png',
        'lidar': '/path/to/before_depth.png'
    }
    after_paths = {
        'rgb': '/path/to/after_rgb.png', 
        'thermal': '/path/to/after_thermal.png',
        'lidar': '/path/to/after_depth.png'
    }
    
    prediction = predict_change(analyzer, before_paths, after_paths)
    
    if prediction:
        print(f"Change Status: {prediction['predicted_class']}")
        print(f"Confidence: {prediction['max_confidence']:.4f}")
"""