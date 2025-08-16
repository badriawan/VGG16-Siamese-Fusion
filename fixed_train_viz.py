# Fixed Training with Visualization (Device & Plotting Issues Resolved)
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from analyzer import MultiModalCorrosionAnalyzer

def train_with_fixed_visualization(data_directory, epochs=10, lr=0.001, batch_size=8):
    """
    Fixed training with visualization - resolves device and plotting issues
    """
    print("="*60)
    print("FIXED TRAINING WITH VISUALIZATION")
    print("="*60)
    
    # Initialize analyzer
    analyzer = MultiModalCorrosionAnalyzer()
    
    # Prepare dataset
    print("Preparing dataset...")
    train_data, val_data = analyzer.prepare_training_dataset(data_directory)
    
    if len(train_data) == 0:
        print("ERROR: No training data found!")
        return None
        
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Train Siamese networks first
    print("\nTraining Siamese networks...")
    try:
        siamese_results = analyzer.train_siamese_networks_for_temporal(
            train_data, val_data, epochs=min(10, epochs//2), lr=lr, batch_size=batch_size
        )
        print("✅ Siamese networks trained successfully")
    except Exception as e:
        print(f"⚠️ Siamese training issue: {e}")
        print("Continuing with change detection training...")
        siamese_results = {}
    
    # Initialize change detection classifier
    print(f"\nTraining change detection classifier...")
    
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
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        analyzer.change_classifier.train()
        epoch_train_loss = 0.0
        train_samples = 0
        
        for sample in train_data:
            try:
                # Extract features with device consistency
                temporal_features = extract_features_safe(analyzer, sample)
                if temporal_features is None:
                    continue
                
                # Prepare input tensor with device consistency
                features_tensor = prepare_feature_tensor_safe(temporal_features, analyzer.device)
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
                
                epoch_train_loss += loss.item()
                train_samples += 1
                
            except Exception as e:
                print(f"Error processing sample {sample.get('sample_id', 'unknown')}: {e}")
                continue
        
        # Calculate average training loss
        avg_train_loss = epoch_train_loss / max(train_samples, 1)
        
        # Validation phase
        analyzer.change_classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sample in val_data:
                try:
                    # Extract features with device consistency
                    temporal_features = extract_features_safe(analyzer, sample)
                    if temporal_features is None:
                        continue
                    
                    # Prepare input tensor with device consistency
                    features_tensor = prepare_feature_tensor_safe(temporal_features, analyzer.device)
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
        
        # Calculate validation metrics
        avg_val_loss = val_loss / max(val_total, 1)
        val_accuracy = val_correct / max(val_total, 1)
        
        # Update history
        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(avg_val_loss)
        history['val_accuracies'].append(val_accuracy)
        history['epochs'].append(epoch + 1)
        
        # Update best accuracy
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
        
        # Print progress
        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.4f}")
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return {
        'analyzer': analyzer,
        'history': history,
        'best_accuracy': best_val_acc,
        'siamese_results': siamese_results
    }

def extract_features_safe(analyzer, sample):
    """Safely extract features with error handling"""
    try:
        features = analyzer.extract_multimodal_features(
            before_paths=sample['before'],
            after_paths=sample['after']
        )
        return features
    except Exception as e:
        print(f"  Feature extraction error: {e}")
        return None

def prepare_feature_tensor_safe(temporal_features, device):
    """Safely prepare feature tensor with device consistency"""
    try:
        feature_tensors = []
        for modality in ['rgb', 'thermal', 'lidar']:
            before_feat = temporal_features[f'{modality}_before']
            after_feat = temporal_features[f'{modality}_after']
            
            # Ensure tensors are on the correct device and proper type
            if not isinstance(before_feat, torch.Tensor):
                before_feat = torch.tensor(before_feat, dtype=torch.float32, device=device)
            else:
                before_feat = before_feat.to(device, dtype=torch.float32)
                
            if not isinstance(after_feat, torch.Tensor):
                after_feat = torch.tensor(after_feat, dtype=torch.float32, device=device)
            else:
                after_feat = after_feat.to(device, dtype=torch.float32)
                
            feature_tensors.extend([before_feat, after_feat])
        
        # Concatenate all features
        features_tensor = torch.cat(feature_tensors).unsqueeze(0)
        return features_tensor
        
    except Exception as e:
        print(f"  Tensor preparation error: {e}")
        return None

def plot_safe_results(results):
    """Plot results with safe array handling"""
    if not results or 'history' not in results:
        print("No training history to plot")
        return
    
    history = results['history']
    epochs = np.array(history['epochs'])
    
    # Create figure with safe plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Training and Validation Loss
    ax1.plot(epochs, history['train_losses'], 'b-', linewidth=2, label='Train Loss', alpha=0.8)
    ax1.plot(epochs, history['val_losses'], 'r-', linewidth=2, label='Val Loss', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy
    ax2.plot(epochs, history['val_accuracies'], 'g-', linewidth=2, label='Val Accuracy')
    ax2.axhline(y=results['best_accuracy'], color='orange', linestyle='--', 
                label=f'Best: {results["best_accuracy"]:.3f}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Learning Progress (Safe)
    if len(history['val_accuracies']) > 1:
        improvement = np.diff([0] + history['val_accuracies'])
        # Ensure matching array lengths
        epochs_subset = epochs[:len(improvement)]
        
        ax3.bar(epochs_subset, improvement, alpha=0.7, color='purple', label='Accuracy Change')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy Improvement')
        ax3.set_title('Learning Progress')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Insufficient data\nfor progress plot', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # Plot 4: Training Summary
    ax4.axis('off')
    
    final_train_loss = history['train_losses'][-1] if history['train_losses'] else 0
    final_val_loss = history['val_losses'][-1] if history['val_losses'] else 0
    
    summary_text = f"""
    TRAINING SUMMARY
    ================
    
    Epochs: {len(epochs)}
    Best Val Accuracy: {results['best_accuracy']:.4f}
    Final Train Loss: {final_train_loss:.4f}
    Final Val Loss: {final_val_loss:.4f}
    
    Status: {'Good' if results['best_accuracy'] > 0.5 else 'Needs Improvement'}
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# Easy usage function
def run_fixed_training(data_directory="/content/drive/MyDrive/S3 UTP/MS2_dataset/temporal_dataset", 
                      epochs=15, lr=0.001):
    """Run training with all fixes applied"""
    print("Running training with device and plotting fixes...")
    
    results = train_with_fixed_visualization(
        data_directory=data_directory,
        epochs=epochs,
        lr=lr,
        batch_size=8
    )
    
    if results:
        print("\nCreating safe plots...")
        plot_safe_results(results)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📊 Best accuracy: {results['best_accuracy']:.4f}")
        
        return results
    else:
        print("❌ Training failed!")
        return None