# VGG16-Siamese-Fusion: Temporal Change Detection

## Updated Objective (v1.0)

**Previous Objective**: Multi-class classification of corrosion severity (no_corrosion, mild, severe)

**NEW OBJECTIVE**: **Temporal Change Detection** - Detect corrosion growth/progression between before/after timepoints using Siamese networks with multi-modal data (RGB, Thermal, LIDAR) for pitting corrosion analysis.

### Key Features:
- **Binary Classification**: Change Detection (0: No Change, 1: Change Detected)
- **Temporal Analysis**: Compares before/after image pairs
- **Multi-modal Fusion**: RGB + Thermal + LIDAR data
- **Siamese Architecture**: Learns similarity/difference representations
- **Backward Compatibility**: Legacy functions maintained

## Modular Structure

This project is now organized into separate modules:

- `VGG16.py`: VGG16 feature extractor
- `siamese.py`: Siamese network architecture
- `loss.py`: Contrastive loss function
- `analyzer.py`: Main analyzer class with all methods
- `main.py`: Main execution script with import statements

## Temporal Dataset Structure

The system now expects temporal dataset with before/after structure:

```
temporal_dataset/
├── sample_001/
│   ├── before/
│   │   ├── rgb.png
│   │   ├── thermal.png
│   │   └── depth.png (LIDAR)
│   ├── after/
│   │   ├── rgb.png
│   │   ├── thermal.png
│   │   └── depth.png (LIDAR)
│   └── label.txt      # 0 = no change, 1 = change detected
├── sample_002/
│   ├── before/
│   ├── after/
│   └── label.txt
└── ...
```

## Usage (NEW Temporal System)

```python
# Import temporal functions
from main import *

# Initialize temporal analyzer
analyzer = MultiModalCorrosionAnalyzer()

# Train temporal change detection system
analyzer, results = train_temporal_change_detection()

# Evaluate temporal system
test_dir = "/path/to/temporal_test_dataset"
evaluation_results = evaluate_temporal_change_detection(analyzer, test_dir)

# Make temporal predictions
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

prediction = analyzer.predict_temporal_change(
    before_paths=before_paths,
    after_paths=after_paths
)

print(f"Change Status: {prediction['predicted_class']}")
print(f"Confidence: {prediction['max_confidence']:.4f}")
```

## Legacy Usage (Backward Compatibility)

```python
# Import all necessary modules
from main import *

# Initialize analyzer
analyzer = MultiModalCorrosionAnalyzer()

# Run legacy training
analyzer, results = safe_training_legacy()

# Run legacy evaluation
evaluation_results = evaluate_and_visualize_results(analyzer, test_data_directory)
```

## Google Colab Usage (Temporal System)

```bash
# Clone repository
!git clone -b v1.0 https://github.com/badriawan/VGG16-Siamese-Fusion.git
%cd /content/VGG16-Siamese-Fusion/

# Test system
!python3 test_temporal.py

# Run temporal training (auto-detects dataset type)
!python main.py
```

### Manual Training Commands:

```python
# For temporal dataset:
analyzer, results = train_temporal_change_detection()

# For legacy dataset:
analyzer, results = safe_training_legacy()

# Auto-detection:
analyzer, results = safe_training()  # Detects and uses appropriate method
```


## File Structure

```
VGG16-Siamese-Fusion/
├── VGG16.py           # VGG16 feature extractor
├── siamese.py         # Siamese network
├── loss.py            # Contrastive loss
├── analyzer.py        # Main analyzer class
├── main.py            # Main script
└── README.md          # This file
```
