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
            train_data, val_data = analyzer.prepare_temporal_training_dataset(data_directory)
            print(f"✓ Dataset preparation completed: {len(train_data)} train, {len(val_data)} val")
            
            # Test visualization
            analyzer.visualize_dataset_distribution(data_directory)
            plt.show()
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
        plt.show()
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

def main():
    """Main function for temporal change detection system"""
    print("="*60)
    print("TEMPORAL CORROSION CHANGE DETECTION ANALYZER")
    print("Updated Objective: Detect corrosion growth progression")
    print("Input: Before/After temporal pairs (RGB, Thermal, LIDAR)")
    print("Output: Binary classification (Change/No Change)")
    print("="*60)
    
    # Test imports first
    if not test_imports():
        print("❌ Import test failed! Please check your file structure.")
        return
    
    print("\nAvailable Options:")
    print("1. Quick test (temporal/legacy auto-detection)")
    print("2. Train temporal change detection system (NEW)")
    print("3. Train legacy classification system")
    print("4. Complete temporal pipeline")
    print("5. Evaluate trained temporal system")
    
    # For demonstration, run quick test
    print("\nRunning automatic test...")
    quick_test()


# Usage examples - uncomment to run
if __name__ == "__main__":
    # Test imports
    test_imports()
    
    # Example 1: Quick test
    # quick_test()
    
    # Example 2: Safe training
    # analyzer, results = safe_training()
    
    # Example 3: Complete training pipeline
    # analyzer, results = train_complete_system()
    
    # Example 4: Siamese networks only
    # analyzer, results = train_siamese_only()

# Manual usage examples:
"""
# Test imports
test_imports()

# Quick test
quick_test()

# Safe training (recommended for first time)
analyzer, results = safe_training()

# Complete training pipeline
analyzer, results = train_complete_system()

# Train only Siamese networks
analyzer, results = train_siamese_only()

# Evaluation after training
evaluation_results = evaluate_and_visualize_results(analyzer, test_data_directory)
"""






