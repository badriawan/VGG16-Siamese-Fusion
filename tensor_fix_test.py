# Test Tensor Fixes
import torch
import os
from analyzer import MultiModalCorrosionAnalyzer

def test_tensor_fixes(data_directory="/content/drive/MyDrive/S3 UTP/MS2_dataset/temporal_dataset"):
    """
    Test the tensor conversion fixes
    """
    print("="*60)
    print("TESTING TENSOR FIXES")
    print("="*60)
    
    try:
        # Initialize analyzer
        analyzer = MultiModalCorrosionAnalyzer()
        print("✅ Analyzer initialized")
        
        # Prepare dataset
        train_data, val_data = analyzer.prepare_training_dataset(data_directory)
        
        if len(train_data) == 0:
            print("❌ No training data found!")
            return False
            
        print(f"✅ Found {len(train_data)} training samples")
        
        # Test feature extraction on first sample
        test_sample = train_data[0]
        print(f"\nTesting sample: {test_sample.get('sample_id', 'unknown')}")
        print(f"Label: {test_sample['change_label']}")
        
        # Extract features
        features = analyzer.extract_multimodal_features(
            before_paths=test_sample['before'],
            after_paths=test_sample['after']
        )
        
        print("\n✅ Feature extraction successful!")
        print("Feature types and shapes:")
        for key, feature in features.items():
            if isinstance(feature, torch.Tensor):
                print(f"  {key}: torch.Tensor {feature.shape} (device: {feature.device})")
            else:
                print(f"  {key}: {type(feature)} {getattr(feature, 'shape', 'no shape')}")
        
        # Test tensor concatenation
        print("\nTesting tensor concatenation...")
        feature_tensors = []
        for modality in ['rgb', 'thermal', 'lidar']:
            before_feat = features[f'{modality}_before']
            after_feat = features[f'{modality}_after']
            
            # Ensure tensors are on the correct device
            if not isinstance(before_feat, torch.Tensor):
                before_feat = torch.tensor(before_feat, device=analyzer.device)
                print(f"  Converted {modality}_before to tensor")
            if not isinstance(after_feat, torch.Tensor):
                after_feat = torch.tensor(after_feat, device=analyzer.device)
                print(f"  Converted {modality}_after to tensor")
                
            feature_tensors.extend([before_feat, after_feat])
            print(f"  {modality}: before {before_feat.shape}, after {after_feat.shape}")
        
        # Test concatenation
        try:
            features_tensor = torch.cat(feature_tensors).unsqueeze(0)
            print(f"✅ Tensor concatenation successful! Shape: {features_tensor.shape}")
            
            # Test model forward pass
            print("\nTesting model forward pass...")
            
            # Initialize simple classifier for testing
            test_classifier = torch.nn.Sequential(
                torch.nn.Linear(features_tensor.shape[1], 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 2)
            ).to(analyzer.device)
            
            with torch.no_grad():
                output = test_classifier(features_tensor)
                print(f"✅ Model forward pass successful! Output shape: {output.shape}")
                print(f"Output values: {output}")
                
            return True
            
        except Exception as e:
            print(f"❌ Tensor concatenation failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_one_epoch(data_directory="/content/drive/MyDrive/S3 UTP/MS2_dataset/temporal_dataset"):
    """
    Test training for one epoch to verify fixes
    """
    print("\n" + "="*60)
    print("TESTING ONE EPOCH TRAINING")
    print("="*60)
    
    try:
        # Initialize analyzer
        analyzer = MultiModalCorrosionAnalyzer()
        
        # Prepare dataset
        train_data, val_data = analyzer.prepare_training_dataset(data_directory)
        
        if len(train_data) == 0:
            print("❌ No training data!")
            return False
            
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        
        # Initialize classifier
        analyzer.change_classifier = torch.nn.Sequential(
            analyzer.fusion_net,
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, 2)
        ).to(analyzer.device)
        
        optimizer = torch.optim.Adam(analyzer.change_classifier.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        print("\nTraining one epoch...")
        
        # Training phase
        analyzer.change_classifier.train()
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        processed_samples = 0
        
        for i, sample in enumerate(train_data[:5]):  # Test first 5 samples
            try:
                print(f"\nProcessing sample {i+1}/5: {sample.get('sample_id', 'unknown')}")
                
                # Extract features
                features = analyzer.extract_multimodal_features(
                    before_paths=sample['before'],
                    after_paths=sample['after']
                )
                
                # Prepare input tensor
                feature_tensors = []
                for modality in ['rgb', 'thermal', 'lidar']:
                    before_feat = features[f'{modality}_before']
                    after_feat = features[f'{modality}_after']
                    
                    # Ensure tensors are on the correct device
                    if not isinstance(before_feat, torch.Tensor):
                        before_feat = torch.tensor(before_feat, device=analyzer.device)
                    if not isinstance(after_feat, torch.Tensor):
                        after_feat = torch.tensor(after_feat, device=analyzer.device)
                        
                    feature_tensors.extend([before_feat, after_feat])
                
                features_tensor = torch.cat(feature_tensors).unsqueeze(0)
                label_tensor = torch.tensor([sample['change_label']], dtype=torch.long).to(analyzer.device)
                
                print(f"  Features shape: {features_tensor.shape}")
                print(f"  Label: {sample['change_label']}")
                
                # Forward pass
                optimizer.zero_grad()
                outputs = analyzer.change_classifier(features_tensor)
                loss = criterion(outputs, label_tensor)
                
                print(f"  Output: {outputs}")
                print(f"  Loss: {loss.item():.4f}")
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                processed_samples += 1
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += 1
                correct_predictions += (predicted == label_tensor).sum().item()
                
                print(f"  Predicted: {predicted.item()}, Actual: {sample['change_label']}")
                print(f"  ✅ Sample processed successfully!")
                
            except Exception as e:
                print(f"  ❌ Error processing sample: {e}")
                continue
        
        # Calculate metrics
        if processed_samples > 0:
            avg_loss = epoch_loss / processed_samples
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            print(f"\n" + "="*40)
            print("EPOCH RESULTS")
            print("="*40)
            print(f"Processed samples: {processed_samples}/{len(train_data[:5])}")
            print(f"Average loss: {avg_loss:.4f}")
            print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
            
            if accuracy > 0:
                print("✅ Training is working! Tensor fixes successful!")
                return True
            else:
                print("⚠️  Training runs but accuracy is 0. May need more epochs or data balancing.")
                return True  # Still successful as no tensor errors
        else:
            print("❌ No samples processed successfully!")
            return False
            
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Easy usage function
def run_tensor_fix_validation():
    """Run complete tensor fix validation"""
    print("Running tensor fix validation...")
    
    # Test 1: Basic tensor operations
    print("\n1. Testing basic tensor operations...")
    test1_passed = test_tensor_fixes()
    
    if not test1_passed:
        print("❌ Basic tensor test failed! Check feature extraction.")
        return False
    
    # Test 2: One epoch training
    print("\n2. Testing one epoch training...")
    test2_passed = test_training_one_epoch()
    
    if test2_passed:
        print("\n🎉 All tensor fix tests passed!")
        print("You can now run full training with confidence!")
        return True
    else:
        print("\n❌ Training test failed. Additional debugging needed.")
        return False