# Training with Real-time Visualization for Google Colab
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import time
from analyzer import MultiModalCorrosionAnalyzer

def train_with_live_visualization(data_directory, epochs=50, lr=0.001, batch_size=8, plot_interval=5):
    """
    Train change detection system with real-time visualization
    """
    print("="*60)
    print("TRAINING WITH LIVE VISUALIZATION")
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
    
    # Train Siamese networks first (simplified for demo)
    print("\nTraining Siamese networks...")
    siamese_results = analyzer.train_siamese_networks_for_temporal(
        train_data, val_data, epochs=20, lr=lr, batch_size=batch_size
    )
    
    # Now train change detection classifier with visualization
    print(f"\nTraining change detection classifier with visualization...")
    
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
    
    # Training history for plotting
    history = {
        'train_losses': [],
        'val_losses': [],
        'val_accuracies': [],
        'epochs': []
    }
    
    # Setup live plotting
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    best_val_acc = 0.0
    
    print(f"Starting training for {epochs} epochs...")
    print("Live plots will update every", plot_interval, "epochs")
    
    for epoch in range(epochs):
        # Training phase
        analyzer.change_classifier.train()
        epoch_train_loss = 0.0
        train_samples = 0
        
        for sample in train_data:
            try:
                # Extract features
                temporal_features = analyzer.extract_multimodal_features(
                    before_paths=sample['before'],
                    after_paths=sample['after']
                )
                
                # Prepare input
                feature_vector = []
                for modality in ['rgb', 'thermal', 'lidar']:
                    feature_vector.extend([
                        temporal_features[f'{modality}_before'],
                        temporal_features[f'{modality}_after']
                    ])
                
                # Convert to tensor
                features_tensor = torch.cat(feature_vector).unsqueeze(0).to(analyzer.device)
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
                    # Extract features
                    temporal_features = analyzer.extract_multimodal_features(
                        before_paths=sample['before'],
                        after_paths=sample['after']
                    )
                    
                    # Prepare input
                    feature_vector = []
                    for modality in ['rgb', 'thermal', 'lidar']:
                        feature_vector.extend([
                            temporal_features[f'{modality}_before'],
                            temporal_features[f'{modality}_after']
                        ])
                    
                    features_tensor = torch.cat(feature_vector).unsqueeze(0).to(analyzer.device)
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
        
        # Update plots every plot_interval epochs
        if (epoch + 1) % plot_interval == 0 or epoch == epochs - 1:
            update_live_plots(axes, history, epoch + 1, best_val_acc)
    
    plt.ioff()
    plt.show()
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return {
        'analyzer': analyzer,
        'history': history,
        'best_accuracy': best_val_acc,
        'siamese_results': siamese_results
    }

def update_live_plots(axes, history, current_epoch, best_acc):
    """Update live training plots"""
    # Clear previous plots
    for ax in axes.flat:
        ax.clear()
    
    epochs = history['epochs']
    
    # Plot 1: Training Loss
    axes[0, 0].plot(epochs, history['train_losses'], 'b-', linewidth=2, label='Train Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Validation Loss
    axes[0, 1].plot(epochs, history['val_losses'], 'r-', linewidth=2, label='Val Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Validation Accuracy
    axes[1, 0].plot(epochs, history['val_accuracies'], 'g-', linewidth=2, label='Val Accuracy')
    axes[1, 0].axhline(y=best_acc, color='orange', linestyle='--', alpha=0.7, label=f'Best: {best_acc:.3f}')
    axes[1, 0].set_title('Validation Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: Combined Loss Comparison
    axes[1, 1].plot(epochs, history['train_losses'], 'b-', linewidth=2, alpha=0.7, label='Train Loss')
    axes[1, 1].plot(epochs, history['val_losses'], 'r-', linewidth=2, alpha=0.7, label='Val Loss')
    axes[1, 1].set_title('Training vs Validation Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Update display
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)

def plot_final_results(results):
    """Plot final training results with detailed analysis"""
    if not results or 'history' not in results:
        print("No training history to plot")
        return
    
    history = results['history']
    
    # Create comprehensive plots
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Loss curves with smoothing
    plt.subplot(2, 3, 1)
    epochs = history['epochs']
    
    # Plot raw data
    plt.plot(epochs, history['train_losses'], alpha=0.3, color='blue', label='Train Loss (raw)')
    plt.plot(epochs, history['val_losses'], alpha=0.3, color='red', label='Val Loss (raw)')
    
    # Plot smoothed data (moving average)
    if len(epochs) > 5:
        window = min(5, len(epochs) // 4)
        train_smooth = np.convolve(history['train_losses'], np.ones(window)/window, mode='valid')
        val_smooth = np.convolve(history['val_losses'], np.ones(window)/window, mode='valid')
        epochs_smooth = epochs[window-1:]
        
        plt.plot(epochs_smooth, train_smooth, linewidth=2, color='blue', label='Train Loss (smooth)')
        plt.plot(epochs_smooth, val_smooth, linewidth=2, color='red', label='Val Loss (smooth)')
    
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curve
    plt.subplot(2, 3, 2)
    plt.plot(epochs, history['val_accuracies'], linewidth=2, color='green', label='Validation Accuracy')
    plt.axhline(y=results['best_accuracy'], color='orange', linestyle='--', 
                label=f'Best: {results["best_accuracy"]:.3f}')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Learning curve analysis
    plt.subplot(2, 3, 3)
    final_train_loss = history['train_losses'][-1] if history['train_losses'] else 0
    final_val_loss = history['val_losses'][-1] if history['val_losses'] else 0
    
    plt.bar(['Train Loss', 'Val Loss'], [final_train_loss, final_val_loss], 
            color=['blue', 'red'], alpha=0.7)
    plt.title('Final Loss Comparison')
    plt.ylabel('Loss')
    
    # Add overfitting analysis
    if final_val_loss > final_train_loss * 1.5:
        plt.text(0.5, max(final_train_loss, final_val_loss) * 0.8, 
                'Possible Overfitting', ha='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Training progress over time
    plt.subplot(2, 3, 4)
    if len(history['val_accuracies']) > 1:
        improvement = np.diff([0] + history['val_accuracies'])
        epochs_diff = epochs[1:len(improvement)+1]  # Match array lengths
        plt.plot(epochs_diff, improvement, linewidth=2, color='purple', label='Accuracy Improvement')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Accuracy Improvement per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy Change')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Insufficient data\nfor improvement plot', 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    # Plot 5: Loss distribution
    plt.subplot(2, 3, 5)
    all_losses = history['train_losses'] + history['val_losses']
    plt.hist(history['train_losses'], bins=15, alpha=0.7, label='Train Loss', color='blue')
    plt.hist(history['val_losses'], bins=15, alpha=0.7, label='Val Loss', color='red')
    plt.title('Loss Distribution')
    plt.xlabel('Loss Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Training summary
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Summary statistics
    summary_text = f"""
    TRAINING SUMMARY
    ==================
    
    Total Epochs: {len(epochs)}
    Best Validation Accuracy: {results['best_accuracy']:.4f}
    Final Train Loss: {final_train_loss:.4f}
    Final Val Loss: {final_val_loss:.4f}
    
    Convergence: {'Good' if final_val_loss < final_train_loss * 2 else 'Check for overfitting'}
    
    Loss Reduction:
    Train: {history['train_losses'][0]:.4f} → {final_train_loss:.4f}
    Val: {history['val_losses'][0]:.4f} → {final_val_loss:.4f}
    """
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print("="*60)
    print("TRAINING ANALYSIS")
    print("="*60)
    print(f"🎯 Best Validation Accuracy: {results['best_accuracy']:.4f}")
    print(f"📉 Final Training Loss: {final_train_loss:.4f}")
    print(f"📊 Final Validation Loss: {final_val_loss:.4f}")
    
    if final_val_loss > final_train_loss * 1.5:
        print("⚠️  Warning: Possible overfitting detected!")
        print("   Consider: reducing epochs, adding regularization, or getting more data")
    else:
        print("✅ Training appears well-balanced")
    
    convergence_epochs = len([acc for acc in history['val_accuracies'][-10:] 
                             if abs(acc - results['best_accuracy']) < 0.01])
    if convergence_epochs >= 8:
        print("🎉 Model appears to have converged well")
    else:
        print("🔄 Model might benefit from more training epochs")

# Usage function for easy calling
def easy_train_with_viz(data_directory="/content/drive/MyDrive/S3 UTP/MS2_dataset/temporal_dataset", 
                        epochs=30, lr=0.001):
    """Easy function to call training with visualization"""
    print("Starting training with live visualization...")
    print("This will show real-time plots during training!")
    
    results = train_with_live_visualization(
        data_directory=data_directory,
        epochs=epochs,
        lr=lr,
        batch_size=8,
        plot_interval=3  # Update plots every 3 epochs
    )
    
    if results:
        print("\nCreating final analysis plots...")
        plot_final_results(results)
        
        return results
    else:
        print("Training failed!")
        return None