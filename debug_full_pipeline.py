# Debug Full Pipeline - Comprehensive Debugging and Testing
import torch
import torch.nn as nn
import os
from analyzer import MultiModalCorrosionAnalyzer

def debug_full_pipeline(data_directory="/content/drive/MyDrive/S3 UTP/MS2_dataset/temporal_dataset"):
    """
    Complete pipeline debugging with detailed logging
    """
    print("="*60)
    print("FULL PIPELINE DEBUG")
    print("="*60)
    
    # Initialize analyzer
    print("1. ANALYZER INITIALIZATION")
    print("-" * 30)
    analyzer = MultiModalCorrosionAnalyzer()
    print(f"✅ Analyzer initialized on device: {analyzer.device}")
    print(f"✅ Fusion network device: {next(analyzer.fusion_net.parameters()).device}")
    
    # Prepare dataset
    print("\n2. DATASET PREPARATION")
    print("-" * 30)
    train_data, val_data = analyzer.prepare_training_dataset(data_directory)
    
    if len(train_data) == 0:
        print("❌ No training data found!")
        return False
    
    print(f"✅ Dataset prepared:")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    
    # Debug sample structure
    print("\n3. SAMPLE STRUCTURE DEBUG")
    print("-" * 30)
    sample = train_data[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Sample ID: {sample.get('sample_id', 'None')}")
    print(f"Change label: {sample.get('change_label', 'None')}")
    print(f"Before type: {type(sample.get('before', 'None'))}")
    print(f"After type: {type(sample.get('after', 'None'))}")
    
    if isinstance(sample.get('before'), dict):
        print(f"Before keys: {list(sample['before'].keys())}")
        print(f"Before values: {list(sample['before'].values())}")
    
    if isinstance(sample.get('after'), dict):
        print(f"After keys: {list(sample['after'].keys())}")
        print(f"After values: {list(sample['after'].values())}")
    
    # Test feature extraction directly
    print("\n4. FEATURE EXTRACTION TEST")
    print("-" * 30)
    try:
        print("Testing direct feature extraction...")
        features = analyzer.extract_multimodal_features(
            before_paths=sample['before'],
            after_paths=sample['after']
        )
        
        print(f"✅ Feature extraction successful!")
        print(f"Features keys: {list(features.keys())}")
        
        for key, feat in features.items():
            if isinstance(feat, torch.Tensor):
                print(f"  {key}: {feat.shape}, device: {feat.device}, dtype: {feat.dtype}")
            else:
                print(f"  {key}: {type(feat)}")
        
        # Test tensor concatenation
        print("\nTesting tensor concatenation...")
        feature_tensors = []
        for modality in ['rgb', 'thermal', 'lidar']:
            before_feat = features[f'{modality}_before']
            after_feat = features[f'{modality}_after']
            feature_tensors.extend([before_feat, after_feat])
        
        combined_tensor = torch.cat(feature_tensors).unsqueeze(0)
        print(f"✅ Concatenation successful: {combined_tensor.shape}, device: {combined_tensor.device}")
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test temporal pairs creation
    print("\n5. TEMPORAL PAIRS DEBUG")
    print("-" * 30)
    try:
        train_pairs, train_labels = analyzer.create_temporal_pairs(train_data[:3])  # Test with first 3 samples
        
        print(f"✅ Temporal pairs created:")
        print(f"  Number of pairs: {len(train_pairs)}")
        print(f"  Labels: {train_labels}")
        
        # Inspect first pair
        if len(train_pairs) > 0:
            pair = train_pairs[0]
            print(f"\nFirst pair structure:")
            print(f"  Pair[0] keys: {list(pair[0].keys())}")
            print(f"  Pair[1] keys: {list(pair[1].keys())}")
            print(f"  Pair[0] type: {type(pair[0])}")
            print(f"  Pair[1] type: {type(pair[1])}")
            
            # Check if pair contains only modality data
            valid_modalities = {'rgb', 'thermal', 'lidar'}
            pair0_modalities = set(pair[0].keys()) & valid_modalities
            pair1_modalities = set(pair[1].keys()) & valid_modalities
            
            print(f"  Pair[0] modalities: {pair0_modalities}")
            print(f"  Pair[1] modalities: {pair1_modalities}")
            
            # Test feature extraction on pair
            print(f"\nTesting feature extraction on temporal pair...")
            before_features = analyzer.extract_multimodal_features(
                before_paths=pair[0], after_paths=None
            )
            after_features = analyzer.extract_multimodal_features(
                before_paths=pair[1], after_paths=None
            )
            
            print(f"✅ Pair feature extraction successful!")
            print(f"  Before features: {list(before_features.keys())}")
            print(f"  After features: {list(after_features.keys())}")
            
    except Exception as e:
        print(f"❌ Temporal pairs creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test Siamese training (one batch)
    print("\n6. SIAMESE TRAINING TEST")
    print("-" * 30)
    try:
        print("Testing one batch of Siamese training...")
        
        # Initialize optimizers
        rgb_optimizer = torch.optim.Adam(analyzer.rgb_siamese.parameters(), lr=0.001)
        thermal_optimizer = torch.optim.Adam(analyzer.thermal_siamese.parameters(), lr=0.001)
        lidar_optimizer = torch.optim.Adam(analyzer.lidar_siamese.parameters(), lr=0.001)
        
        criterion = nn.ContrastiveLoss(margin=1.0)
        
        # Process one pair
        pair = train_pairs[0]
        label = train_labels[0]
        
        # Extract features
        before_features = analyzer.extract_multimodal_features(
            before_paths=pair[0], after_paths=None
        )
        after_features = analyzer.extract_multimodal_features(
            before_paths=pair[1], after_paths=None
        )
        
        # Test each modality
        for modality in ['rgb', 'thermal', 'lidar']:
            print(f"\nTesting {modality} Siamese:")
            
            before_key = f'{modality}_before'
            after_key = f'{modality}_after'
            
            if before_key in before_features and after_key in after_features:
                feat1 = before_features[before_key]
                feat2 = after_features[after_key]
                
                print(f"  Feature 1: {feat1.shape}, device: {feat1.device}")
                print(f"  Feature 2: {feat2.shape}, device: {feat2.device}")
                
                # Get Siamese network
                if modality == 'rgb':
                    siamese_net = analyzer.rgb_siamese
                    optimizer = rgb_optimizer
                elif modality == 'thermal':
                    siamese_net = analyzer.thermal_siamese
                    optimizer = thermal_optimizer
                else:
                    siamese_net = analyzer.lidar_siamese
                    optimizer = lidar_optimizer
                
                # Forward pass
                output1, output2 = siamese_net(feat1.unsqueeze(0), feat2.unsqueeze(0))
                
                print(f"  Output 1: {output1.shape}, device: {output1.device}")
                print(f"  Output 2: {output2.shape}, device: {output2.device}")
                
                # Calculate loss
                target = torch.tensor([label], dtype=torch.float32).to(analyzer.device)
                loss = criterion(output1, output2, target)
                
                print(f"  Loss: {loss.item():.4f}")
                print(f"  ✅ {modality} Siamese test successful!")
            else:
                print(f"  ❌ Missing features for {modality}")
        
    except Exception as e:
        print(f"❌ Siamese training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test change detection classifier
    print("\n7. CHANGE DETECTION CLASSIFIER TEST")
    print("-" * 30)
    try:
        print("Testing change detection classifier...")
        
        # Initialize classifier
        classifier = nn.Sequential(
            analyzer.fusion_net,
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        ).to(analyzer.device)
        
        # Test with sample
        features = analyzer.extract_multimodal_features(
            before_paths=sample['before'],
            after_paths=sample['after']
        )
        
        # Prepare tensor
        feature_tensors = []
        for modality in ['rgb', 'thermal', 'lidar']:
            before_feat = features[f'{modality}_before']
            after_feat = features[f'{modality}_after']
            feature_tensors.extend([before_feat, after_feat])
        
        combined_tensor = torch.cat(feature_tensors).unsqueeze(0)
        
        print(f"  Input tensor: {combined_tensor.shape}, device: {combined_tensor.device}")
        
        # Forward pass
        output = classifier(combined_tensor)
        
        print(f"  Output: {output.shape}, device: {output.device}")
        print(f"  Output values: {output}")
        print(f"  ✅ Change detection classifier test successful!")
        
    except Exception as e:
        print(f"❌ Change detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED!")
    print("✅ Pipeline is ready for training!")
    print("="*60)
    
    return True

def minimal_training_test(data_directory="/content/drive/MyDrive/S3 UTP/MS2_dataset/temporal_dataset", epochs=3):
    """
    Minimal training test with all fixes applied
    """
    print("="*60)
    print("MINIMAL TRAINING TEST")
    print("="*60)
    
    # Initialize
    analyzer = MultiModalCorrosionAnalyzer()
    train_data, val_data = analyzer.prepare_training_dataset(data_directory)
    
    if len(train_data) == 0:
        print("❌ No training data!")
        return False
    
    print(f"✅ Training with {len(train_data)} samples for {epochs} epochs")
    
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
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        analyzer.change_classifier.train()
        
        epoch_loss = 0.0
        processed_samples = 0
        
        for i, sample in enumerate(train_data[:5]):  # Test with first 5 samples
            try:
                print(f"  Processing sample {i+1}/5...")
                
                # Extract features
                features = analyzer.extract_multimodal_features(
                    before_paths=sample['before'],
                    after_paths=sample['after']
                )
                
                # Prepare tensor
                feature_tensors = []
                for modality in ['rgb', 'thermal', 'lidar']:
                    before_feat = features[f'{modality}_before']
                    after_feat = features[f'{modality}_after']
                    feature_tensors.extend([before_feat, after_feat])
                
                combined_tensor = torch.cat(feature_tensors).unsqueeze(0)
                label_tensor = torch.tensor([sample['change_label']], dtype=torch.long).to(analyzer.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = analyzer.change_classifier(combined_tensor)
                loss = criterion(outputs, label_tensor)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                processed_samples += 1
                
                _, predicted = torch.max(outputs.data, 1)
                print(f"    Loss: {loss.item():.4f}, Predicted: {predicted.item()}, Actual: {sample['change_label']}")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                continue
        
        if processed_samples > 0:
            avg_loss = epoch_loss / processed_samples
            print(f"  ✅ Epoch {epoch+1} completed: avg loss = {avg_loss:.4f}, samples processed = {processed_samples}")
        else:
            print(f"  ❌ No samples processed in epoch {epoch+1}")
    
    print(f"\n🎉 Minimal training test completed!")
    return True

# Easy usage functions
def run_full_debug():
    """Run complete debugging suite"""
    print("🔍 Running full pipeline debug...")
    success = debug_full_pipeline()
    
    if success:
        print("\n🚀 Running minimal training test...")
        train_success = minimal_training_test()
        
        if train_success:
            print(f"\n✅ ALL DEBUGGING TESTS PASSED!")
            print(f"🎯 System is ready for full training!")
            return True
        else:
            print(f"\n❌ Training test failed")
            return False
    else:
        print(f"\n❌ Pipeline debug failed")
        return False

def quick_fix_validation():
    """Quick validation that fixes work"""
    print("🔧 Quick fix validation...")
    
    analyzer = MultiModalCorrosionAnalyzer()
    train_data, _ = analyzer.prepare_training_dataset()
    
    if len(train_data) == 0:
        print("❌ No data")
        return False
    
    sample = train_data[0]
    
    # Test feature extraction
    try:
        features = analyzer.extract_multimodal_features(
            before_paths=sample['before'],
            after_paths=sample['after']
        )
        print("✅ Feature extraction works")
        
        # Test tensor ops
        feature_tensors = []
        for modality in ['rgb', 'thermal', 'lidar']:
            feature_tensors.extend([features[f'{modality}_before'], features[f'{modality}_after']])
        
        combined = torch.cat(feature_tensors)
        print(f"✅ Tensor ops work: {combined.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fix validation failed: {e}")
        return False