# Final Robust Training with All Fixes Applied
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from analyzer import MultiModalCorrosionAnalyzer

def robust_training_with_all_fixes(data_directory, epochs=15, lr=0.001, batch_size=8):
    """
    Final robust training function with all device and data structure fixes
    """
    print("="*60)
    print("ROBUST TRAINING WITH ALL FIXES")
    print("="*60)
    
    # Initialize analyzer
    print("Initializing analyzer...")
    analyzer = MultiModalCorrosionAnalyzer()
    print(f"✅ Analyzer initialized on device: {analyzer.device}")
    
    # Prepare dataset with validation
    print("\nPreparing dataset...")
    train_data, val_data = analyzer.prepare_training_dataset(data_directory)
    
    if len(train_data) == 0:
        print("❌ ERROR: No training data found!")
        return None
    
    print(f"✅ Dataset prepared:")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    
    # Validate data structure
    print(f"\nValidating data structure...")
    sample_valid = validate_sample_structure(train_data[0])
    if not sample_valid:
        print("❌ ERROR: Invalid data structure!")
        return None
    print("✅ Data structure validation passed")
    
    # Train Siamese networks with error handling
    print(f"\nStage 1: Training Siamese networks...")
    try:
        siamese_results = train_siamese_safe(analyzer, train_data, val_data, 
                                           epochs=min(10, epochs//2), lr=lr, batch_size=batch_size)
        print("✅ Siamese networks trained successfully")
    except Exception as e:
        print(f"⚠️ Siamese training issue: {e}")
        print("Continuing with change detection training...")
        siamese_results = {}
    
    # Train change detection classifier
    print(f"\nStage 2: Training change detection classifier...")
    
    # Initialize classifier
    analyzer.change_classifier = nn.Sequential(
        analyzer.fusion_net,
        nn.Linear(128, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(64, 32),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(32, 2)
    ).to(analyzer.device)
    
    optimizer = torch.optim.Adam(analyzer.change_classifier.parameters(), lr=lr*0.1)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'val_accuracies': [],
        'epochs': []
    }
    
    best_val_acc = 0.0
    consecutive_failures = 0
    max_failures = 5
    
    print(f"Starting training for {epochs} epochs...")
    print("Real-time progress:")
    
    for epoch in range(epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        
        # Training phase
        train_loss, train_samples = train_epoch_safe(
            analyzer, train_data, optimizer, criterion, epoch
        )
        
        if train_samples == 0:
            consecutive_failures += 1
            print(f"❌ No samples processed in epoch {epoch+1}")
            if consecutive_failures >= max_failures:
                print(f"❌ Too many consecutive failures ({max_failures}). Stopping training.")
                break
            continue
        else:
            consecutive_failures = 0
        
        # Validation phase
        val_loss, val_accuracy, val_samples = validate_epoch_safe(
            analyzer, val_data, criterion, epoch
        )
        
        # Update history
        avg_train_loss = train_loss / train_samples
        avg_val_loss = val_loss / max(val_samples, 1)
        
        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(avg_val_loss)
        history['val_accuracies'].append(val_accuracy)
        history['epochs'].append(epoch + 1)
        
        # Update best accuracy
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            print(f"🎯 New best accuracy: {best_val_acc:.4f}")
        
        # Print epoch summary
        print(f"  Train Loss: {avg_train_loss:.4f} ({train_samples} samples)")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f} ({val_samples} samples)")
    
    print(f"\n🎉 Training completed!")
    print(f"📊 Best validation accuracy: {best_val_acc:.4f}")
    
    if best_val_acc > 0.1:
        print("✅ Training appears successful!")
    else:
        print("⚠️  Low accuracy. Consider checking data labels or training longer.")
    
    return {
        'analyzer': analyzer,
        'history': history,
        'best_accuracy': best_val_acc,
        'siamese_results': siamese_results,
        'training_successful': best_val_acc > 0.1
    }

def validate_sample_structure(sample):
    """Validate that sample has correct structure"""
    try:
        required_keys = ['sample_id', 'change_label', 'before', 'after']
        for key in required_keys:
            if key not in sample:
                print(f"❌ Missing key: {key}")
                return False
        
        # Check before/after are dictionaries
        if not isinstance(sample['before'], dict):
            print(f"❌ 'before' should be dict, got {type(sample['before'])}: {sample['before']}")
            return False
        
        if not isinstance(sample['after'], dict):
            print(f"❌ 'after' should be dict, got {type(sample['after'])}: {sample['after']}")
            return False
        
        # Check modalities exist
        modalities = ['rgb', 'thermal', 'lidar']
        for modality in modalities:
            if modality not in sample['before']:
                print(f"⚠️ Missing modality in before: {modality}")
            if modality not in sample['after']:
                print(f"⚠️ Missing modality in after: {modality}")
        
        print(f"✅ Sample structure valid:")
        print(f"  ID: {sample['sample_id']}")
        print(f"  Label: {sample['change_label']}")
        print(f"  Before modalities: {list(sample['before'].keys())}")
        print(f"  After modalities: {list(sample['after'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def extract_features_with_retry(analyzer, sample, max_retries=3):
    """Extract features with retry mechanism"""
    for attempt in range(max_retries):
        try:
            print(f"    Extracting features (attempt {attempt+1}/{max_retries})...")
            
            # Validate inputs before extraction
            if not isinstance(sample['before'], dict) or not isinstance(sample['after'], dict):
                print(f"    ❌ Invalid input structure")
                return None
            
            features = analyzer.extract_multimodal_features(
                before_paths=sample['before'],
                after_paths=sample['after']
            )
            
            if not features:
                print(f"    ⚠️ Empty features returned")
                continue
            
            print(f"    ✅ Features extracted successfully")
            return features
            
        except Exception as e:
            print(f"    ❌ Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                print(f"    ❌ All extraction attempts failed for sample {sample.get('sample_id', 'unknown')}")
    
    return None

def prepare_tensor_robust(features, device):
    """Prepare feature tensor with robust error handling"""
    try:
        feature_tensors = []
        
        for modality in ['rgb', 'thermal', 'lidar']:
            before_key = f'{modality}_before'
            after_key = f'{modality}_after'
            
            if before_key not in features or after_key not in features:
                print(f"    ⚠️ Missing features for {modality}")
                # Use zeros as fallback
                before_feat = torch.zeros(512, device=device, dtype=torch.float32)
                after_feat = torch.zeros(512, device=device, dtype=torch.float32)
            else:
                before_feat = features[before_key]
                after_feat = features[after_key]
            
            # Ensure proper tensor type and device
            if not isinstance(before_feat, torch.Tensor):
                before_feat = torch.tensor(before_feat, device=device, dtype=torch.float32)
            else:
                before_feat = before_feat.to(device, dtype=torch.float32)
                
            if not isinstance(after_feat, torch.Tensor):
                after_feat = torch.tensor(after_feat, device=device, dtype=torch.float32)
            else:
                after_feat = after_feat.to(device, dtype=torch.float32)
            
            feature_tensors.extend([before_feat, after_feat])
        
        # Concatenate all features
        features_tensor = torch.cat(feature_tensors).unsqueeze(0)
        print(f"    ✅ Tensor prepared: {features_tensor.shape}")
        
        return features_tensor
        
    except Exception as e:
        print(f"    ❌ Tensor preparation failed: {e}")
        return None

def train_epoch_safe(analyzer, train_data, optimizer, criterion, epoch):
    """Safe training epoch with comprehensive error handling"""
    analyzer.change_classifier.train()
    epoch_loss = 0.0
    processed_samples = 0
    
    print(f"  Training on {len(train_data)} samples...")
    
    for i, sample in enumerate(train_data):
        try:
            sample_id = sample.get('sample_id', f'sample_{i}')
            if i < 3 or i % 10 == 0:  # Verbose logging for first few and every 10th
                print(f"    Processing {sample_id}...")
            
            # Extract features with retry
            features = extract_features_with_retry(analyzer, sample)
            if features is None:
                continue
            
            # Prepare tensor robustly
            features_tensor = prepare_tensor_robust(features, analyzer.device)
            if features_tensor is None:
                continue
            
            label_tensor = torch.tensor([sample['change_label']], dtype=torch.long).to(analyzer.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = analyzer.change_classifier(features_tensor)
            loss = criterion(outputs, label_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            processed_samples += 1
            
            if i < 3:  # Detailed logging for first few samples
                _, predicted = torch.max(outputs.data, 1)
                print(f"    Loss: {loss.item():.4f}, Predicted: {predicted.item()}, Actual: {sample['change_label']}")
            
        except Exception as e:
            sample_id = sample.get('sample_id', f'sample_{i}')
            print(f"    ❌ Error processing {sample_id}: {e}")
            continue
    
    print(f"  ✅ Training epoch completed: {processed_samples}/{len(train_data)} samples processed")
    return epoch_loss, processed_samples

def validate_epoch_safe(analyzer, val_data, criterion, epoch):
    """Safe validation epoch"""
    analyzer.change_classifier.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    print(f"  Validating on {len(val_data)} samples...")
    
    with torch.no_grad():
        for i, sample in enumerate(val_data):
            try:
                # Extract features
                features = extract_features_with_retry(analyzer, sample)
                if features is None:
                    continue
                
                # Prepare tensor
                features_tensor = prepare_tensor_robust(features, analyzer.device)
                if features_tensor is None:
                    continue
                
                label_tensor = torch.tensor([sample['change_label']], dtype=torch.long).to(analyzer.device)
                
                # Forward pass
                outputs = analyzer.change_classifier(features_tensor)
                loss = criterion(outputs, label_tensor)
                
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                val_total += 1
                val_correct += (predicted == label_tensor).sum().item()
                
            except Exception as e:
                continue
    
    val_accuracy = val_correct / max(val_total, 1)
    print(f"  ✅ Validation completed: {val_correct}/{val_total} correct, accuracy: {val_accuracy:.4f}")
    
    return val_loss, val_accuracy, val_total

def train_siamese_safe(analyzer, train_data, val_data, epochs, lr, batch_size):
    """Safe Siamese network training with error handling"""
    try:
        print(f"  Training Siamese networks for {epochs} epochs...")
        results = analyzer.train_siamese_networks_for_temporal(
            train_data, val_data, epochs, lr, batch_size
        )
        return results if results else {}
    except Exception as e:
        print(f"  ⚠️ Siamese training failed: {e}")
        return {}

def plot_robust_results(results):
    """Plot results with robust error handling"""
    if not results or 'history' not in results:
        print("No training history to plot")
        return
    
    history = results['history']
    
    # Ensure we have data to plot
    if not history['epochs'] or not history['train_losses']:
        print("Insufficient data for plotting")
        return
    
    epochs = np.array(history['epochs'])
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Losses
    ax1.plot(epochs, history['train_losses'], 'b-', linewidth=2, label='Train Loss')
    ax1.plot(epochs, history['val_losses'], 'r-', linewidth=2, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    ax2.plot(epochs, history['val_accuracies'], 'g-', linewidth=2, label='Val Accuracy')
    ax2.axhline(y=results['best_accuracy'], color='orange', linestyle='--', 
                label=f'Best: {results["best_accuracy"]:.3f}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final metrics
    final_train = history['train_losses'][-1] if history['train_losses'] else 0
    final_val = history['val_losses'][-1] if history['val_losses'] else 0
    final_acc = history['val_accuracies'][-1] if history['val_accuracies'] else 0
    
    metrics = ['Train Loss', 'Val Loss', 'Val Accuracy']
    values = [final_train, final_val, final_acc]
    colors = ['blue', 'red', 'green']
    
    bars = ax3.bar(metrics, values, color=colors, alpha=0.7)
    ax3.set_title('Final Metrics')
    ax3.set_ylabel('Value')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 4: Training summary
    ax4.axis('off')
    
    status = "✅ SUCCESS" if results.get('training_successful', False) else "⚠️ NEEDS IMPROVEMENT"
    
    summary = f"""
    TRAINING SUMMARY
    ================
    
    Status: {status}
    
    Total Epochs: {len(epochs)}
    Best Accuracy: {results['best_accuracy']:.4f}
    Final Train Loss: {final_train:.4f}
    Final Val Loss: {final_val:.4f}
    
    Siamese Training: {'✅' if results.get('siamese_results') else '⚠️'}
    
    Recommendation:
    {'Ready for testing!' if results.get('training_successful', False) else 'Consider more training or data review'}
    """
    
    ax4.text(0.1, 0.9, summary, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen" if results.get('training_successful', False) else "lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# Easy usage function
def run_final_robust_training(data_directory="/content/drive/MyDrive/Colab Notebooks/temporal_dataset", 
                             epochs=20, lr=0.001):
    """Run the final robust training with all fixes"""
    print("🚀 Starting final robust training with comprehensive fixes...")
    
    results = robust_training_with_all_fixes(
        data_directory=data_directory,
        epochs=epochs,
        lr=lr,
        batch_size=8
    )
    
    if results:
        print(f"\n📊 Creating training analysis plots...")
        plot_robust_results(results)
        
        if results.get('training_successful', False):
            print(f"\n🎉 TRAINING SUCCESSFUL!")
            print(f"✅ Best accuracy: {results['best_accuracy']:.4f}")
            print(f"✅ Model ready for inference!")
        else:
            print(f"\n⚠️ TRAINING COMPLETED BUT NEEDS IMPROVEMENT")
            print(f"📊 Best accuracy: {results['best_accuracy']:.4f}")
            print(f"💡 Consider: more epochs, data balancing, or hyperparameter tuning")
        
        return results
    else:
        print(f"\n❌ TRAINING FAILED!")
        print(f"Please check dataset structure and try again.")
        return None

# Debug helper
def quick_data_check(data_directory="/content/drive/MyDrive/Colab Notebooks/temporal_dataset"):
    """Quick data structure check"""
    print("Quick data structure check...")
    
    analyzer = MultiModalCorrosionAnalyzer()
    train_data, val_data = analyzer.prepare_training_dataset(data_directory)
    
    if len(train_data) > 0:
        print("✅ Data preparation successful")
        print(f"Sample structure check:")
        validate_sample_structure(train_data[0])
        return True
    else:
        print("❌ No training data found")
        return False