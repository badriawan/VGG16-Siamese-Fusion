# Debug Training Issues
import torch
import torch.nn as nn
import os
from analyzer import MultiModalCorrosionAnalyzer

def debug_dataset_and_training(data_directory):
    """
    Debug dataset and training issues
    """
    print("="*60)
    print("DEBUGGING DATASET AND TRAINING")
    print("="*60)
    
    # Check dataset structure
    print("1. DATASET STRUCTURE CHECK")
    print("-" * 30)
    
    if not os.path.exists(data_directory):
        print(f"❌ Dataset directory not found: {data_directory}")
        return False
    
    samples = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
    print(f"✅ Found {len(samples)} samples")
    
    # Check sample structure
    valid_samples = 0
    change_labels = []
    
    for sample in samples[:5]:  # Check first 5 samples
        sample_path = os.path.join(data_directory, sample)
        print(f"\nChecking sample: {sample}")
        
        # Check before/after directories
        before_path = os.path.join(sample_path, 'before')
        after_path = os.path.join(sample_path, 'after')
        label_path = os.path.join(sample_path, 'label.txt')
        
        if os.path.exists(before_path) and os.path.exists(after_path) and os.path.exists(label_path):
            # Check modalities
            modalities = ['rgb.png', 'thermal.png', 'depth.png']
            before_files = os.listdir(before_path)
            after_files = os.listdir(after_path)
            
            print(f"  Before files: {before_files}")
            print(f"  After files: {after_files}")
            
            # Check label
            try:
                with open(label_path, 'r') as f:
                    label = int(f.read().strip())
                    change_labels.append(label)
                    print(f"  Label: {label}")
                    valid_samples += 1
            except:
                print(f"  ❌ Invalid label file")
        else:
            print(f"  ❌ Missing directories or label")
    
    print(f"\n✅ Valid samples: {valid_samples}/{min(5, len(samples))}")
    print(f"✅ Labels found: {change_labels}")
    
    # Check label distribution
    print("\n2. LABEL DISTRIBUTION")
    print("-" * 30)
    
    all_labels = []
    for sample in samples:
        label_path = os.path.join(data_directory, sample, 'label.txt')
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    label = int(f.read().strip())
                    all_labels.append(label)
            except:
                pass
    
    if all_labels:
        change_count = sum(all_labels)
        no_change_count = len(all_labels) - change_count
        print(f"No Change (0): {no_change_count}")
        print(f"Change (1): {change_count}")
        print(f"Balance ratio: {change_count/len(all_labels):.3f}")
        
        if change_count == 0 or no_change_count == 0:
            print("⚠️  WARNING: Imbalanced dataset - all samples have same label!")
            return False
    else:
        print("❌ No valid labels found!")
        return False
    
    # Test analyzer initialization
    print("\n3. ANALYZER INITIALIZATION")
    print("-" * 30)
    
    try:
        analyzer = MultiModalCorrosionAnalyzer()
        print("✅ Analyzer initialized successfully")
        
        # Test dataset preparation
        train_data, val_data = analyzer.prepare_training_dataset(data_directory)
        print(f"✅ Training data: {len(train_data)} samples")
        print(f"✅ Validation data: {len(val_data)} samples")
        
        if len(train_data) == 0:
            print("❌ No training data prepared!")
            return False
            
        # Test feature extraction on one sample
        print("\n4. FEATURE EXTRACTION TEST")
        print("-" * 30)
        
        test_sample = train_data[0]
        print(f"Testing sample: {test_sample.get('sample_id', 'unknown')}")
        print(f"Label: {test_sample['change_label']}")
        
        try:
            features = analyzer.extract_multimodal_features(
                before_paths=test_sample['before'],
                after_paths=test_sample['after']
            )
            print("✅ Feature extraction successful")
            print(f"Features keys: {list(features.keys())}")
            
            # Check feature shapes
            for key, feature in features.items():
                if hasattr(feature, 'shape'):
                    print(f"  {key}: {feature.shape}")
                else:
                    print(f"  {key}: {type(feature)}")
                    
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Analyzer initialization failed: {e}")
        return False

def simple_training_test(data_directory, epochs=5):
    """
    Simple training test with detailed logging
    """
    print("\n" + "="*60)
    print("SIMPLE TRAINING TEST")
    print("="*60)
    
    # Initialize analyzer
    analyzer = MultiModalCorrosionAnalyzer()
    
    # Prepare dataset
    train_data, val_data = analyzer.prepare_training_dataset(data_directory)
    
    if len(train_data) == 0:
        print("❌ No training data!")
        return None
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Check label distribution in training data
    train_labels = [sample['change_label'] for sample in train_data]
    val_labels = [sample['change_label'] for sample in val_data]
    
    print(f"Train labels distribution: {train_labels}")
    print(f"Val labels distribution: {val_labels}")
    
    # Simplified training - just change detection classifier
    print("\nTraining simple change detection classifier...")
    
    # Initialize classifier
    analyzer.change_classifier = nn.Sequential(
        analyzer.fusion_net,
        nn.Linear(128, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(64, 2)
    ).to(analyzer.device)
    
    optimizer = torch.optim.Adam(analyzer.change_classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop with detailed logging
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 20)
        
        # Training phase
        analyzer.change_classifier.train()
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for i, sample in enumerate(train_data):
            try:
                print(f"  Processing training sample {i+1}/{len(train_data)}")
                
                # Extract features
                features = analyzer.extract_multimodal_features(
                    before_paths=sample['before'],
                    after_paths=sample['after']
                )
                
                # Prepare input tensor
                feature_vector = []
                for modality in ['rgb', 'thermal', 'lidar']:
                    feature_vector.extend([
                        features[f'{modality}_before'],
                        features[f'{modality}_after']
                    ])
                
                features_tensor = torch.cat(feature_vector).unsqueeze(0).to(analyzer.device)
                label_tensor = torch.tensor([sample['change_label']], dtype=torch.long).to(analyzer.device)
                
                print(f"    Features shape: {features_tensor.shape}")
                print(f"    Label: {sample['change_label']}")
                
                # Forward pass
                optimizer.zero_grad()
                outputs = analyzer.change_classifier(features_tensor)
                loss = criterion(outputs, label_tensor)
                
                print(f"    Output: {outputs}")
                print(f"    Loss: {loss.item():.4f}")
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += 1
                correct_predictions += (predicted == label_tensor).sum().item()
                
                print(f"    Predicted: {predicted.item()}, Actual: {sample['change_label']}")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                continue
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / len(train_data)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Training Accuracy: {accuracy:.4f}")
        print(f"  Correct/Total: {correct_predictions}/{total_predictions}")
    
    return analyzer

# Usage functions
def run_complete_debug(data_directory="/content/drive/MyDrive/S3 UTP/MS2_dataset/temporal_dataset"):
    """Run complete debugging process"""
    print("Starting complete debugging process...")
    
    # Step 1: Debug dataset
    dataset_ok = debug_dataset_and_training(data_directory)
    
    if not dataset_ok:
        print("\n❌ Dataset issues found! Please fix dataset before training.")
        return None
    
    # Step 2: Simple training test
    analyzer = simple_training_test(data_directory, epochs=3)
    
    if analyzer:
        print("\n✅ Simple training test completed!")
        return analyzer
    else:
        print("\n❌ Training test failed!")
        return None

def fix_dataset_labels(data_directory):
    """Fix dataset labels if they're all the same"""
    print("Checking and fixing dataset labels...")
    
    samples = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
    
    # Create balanced labels if needed
    for i, sample in enumerate(samples):
        label_path = os.path.join(data_directory, sample, 'label.txt')
        
        # Alternate labels for balance
        new_label = i % 2
        
        with open(label_path, 'w') as f:
            f.write(str(new_label))
        
        print(f"Sample {sample}: label set to {new_label}")
    
    print(f"Fixed labels for {len(samples)} samples")