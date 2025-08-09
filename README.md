# VGG16-Siamese-Fusion

## Modular Structure

This project is now organized into separate modules:

- `VGG16.py`: VGG16 feature extractor
- `siamese.py`: Siamese network architecture
- `loss.py`: Contrastive loss function
- `analyzer.py`: Main analyzer class with all methods
- `main.py`: Main execution script with import statements

## Usage

```python
# Import all necessary modules
from main import *

# Initialize analyzer
analyzer = MultiModalCorrosionAnalyzer()

# Run training
analyzer, results = safe_training()

# Run evaluation
evaluation_results = evaluate_and_visualize_results(analyzer, test_data_directory)
```

## Google Collab Usage
```
!git clone -b v1.0 https://github.com/badriawan/VGG16-Siamese-Fusion.git

%cd /content/VGG16-Siamese-Fusion/
!python main.py
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
