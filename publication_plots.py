# Publication-Quality Plots for Journal Submission
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'patch.linewidth': 0.5,
    'patch.facecolor': 'lightblue',
    'patch.edgecolor': 'black'
})

def create_training_curves_figure(results, save_path='/content/training_curves.png'):
    """
    Create publication-quality training curves figure
    Suitable for: Results section of journal paper
    """
    if not results or 'history' not in results:
        print("No training history available")
        return None
        
    history = results['history']
    epochs = np.array(history['epochs'])
    
    # Create figure with 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Color scheme for publication
    colors = {
        'train': '#1f77b4',  # Blue
        'val': '#ff7f0e',    # Orange
        'acc': '#2ca02c',    # Green
        'best': '#d62728'    # Red
    }
    
    # (a) Training and Validation Loss
    ax1.plot(epochs, history['train_losses'], color=colors['train'], 
             linewidth=2.5, label='Training Loss', alpha=0.8)
    ax1.plot(epochs, history['val_losses'], color=colors['val'], 
             linewidth=2.5, label='Validation Loss', alpha=0.8)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('(a) Training and Validation Loss')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Add loss values as text
    final_train_loss = history['train_losses'][-1]
    final_val_loss = history['val_losses'][-1]
    ax1.text(0.65, 0.95, f'Final Train Loss: {final_train_loss:.4f}\nFinal Val Loss: {final_val_loss:.4f}', 
             transform=ax1.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    # (b) Validation Accuracy
    ax2.plot(epochs, history['val_accuracies'], color=colors['acc'], 
             linewidth=2.5, label='Validation Accuracy')
    ax2.axhline(y=results['best_accuracy'], color=colors['best'], 
                linestyle='--', linewidth=2, alpha=0.8,
                label=f'Best Accuracy: {results["best_accuracy"]:.4f}')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('(b) Validation Accuracy')
    ax2.set_ylim(0, 1.05)
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    # (c) Learning Rate Analysis (Convergence)
    # Calculate moving average for smoothness
    window_size = max(3, len(epochs) // 10)
    if len(history['val_accuracies']) >= window_size:
        acc_smooth = np.convolve(history['val_accuracies'], 
                                np.ones(window_size)/window_size, mode='valid')
        epochs_smooth = epochs[window_size-1:]
        
        ax3.plot(epochs, history['val_accuracies'], color=colors['acc'], 
                alpha=0.3, linewidth=1, label='Raw Accuracy')
        ax3.plot(epochs_smooth, acc_smooth, color=colors['acc'], 
                linewidth=2.5, label='Smoothed Accuracy')
    else:
        ax3.plot(epochs, history['val_accuracies'], color=colors['acc'], 
                linewidth=2.5, label='Validation Accuracy')
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('(c) Convergence Analysis')
    ax3.legend(frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)
    
    # (d) Training Efficiency
    # Calculate accuracy improvement rate
    acc_diff = np.diff([0] + history['val_accuracies'])
    
    ax4.bar(epochs[1:], acc_diff, color=colors['acc'], alpha=0.7, 
           width=0.8, label='Accuracy Improvement')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy Improvement')
    ax4.set_title('(d) Learning Progress')
    ax4.legend(frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Publication-quality training curves saved to: {save_path}")
    return fig

def create_architecture_diagram(save_path='/content/architecture.png'):
    """
    Create VGG16-Siamese Fusion architecture diagram
    Suitable for: Methodology section
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Colors for different components
    colors = {
        'input': '#E8F4FD',
        'vgg': '#B8E6B8',
        'siamese': '#FFE4B5',
        'fusion': '#FFB6C1',
        'output': '#D8BFD8'
    }
    
    # Title
    ax.text(10, 11.5, 'VGG16-Siamese Fusion Network Architecture', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Input layer - Before images
    before_y = 9
    inputs_before = ['RGB', 'Thermal', 'LiDAR']
    for i, inp in enumerate(inputs_before):
        x = 1 + i * 2
        rect = Rectangle((x-0.4, before_y-0.3), 0.8, 0.6, 
                        facecolor=colors['input'], edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x, before_y, f'{inp}\n(Before)', ha='center', va='center', fontsize=10)
    
    # Input layer - After images  
    after_y = 7
    for i, inp in enumerate(inputs_before):
        x = 1 + i * 2
        rect = Rectangle((x-0.4, after_y-0.3), 0.8, 0.6, 
                        facecolor=colors['input'], edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x, after_y, f'{inp}\n(After)', ha='center', va='center', fontsize=10)
    
    # VGG16 Feature Extractors
    vgg_y = 8
    for i in range(3):
        x = 8 + i * 2
        rect = Rectangle((x-0.6, vgg_y-0.8), 1.2, 1.6, 
                        facecolor=colors['vgg'], edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x, vgg_y, f'VGG16\nExtractor\n{inputs_before[i]}', 
               ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Siamese Networks
    siamese_y = 5.5
    for i in range(3):
        x = 8 + i * 2
        rect = Rectangle((x-0.6, siamese_y-0.5), 1.2, 1, 
                        facecolor=colors['siamese'], edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x, siamese_y, f'Siamese\nNetwork\n{inputs_before[i]}', 
               ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Temporal Fusion Network
    fusion_x = 15
    fusion_y = 6.5
    rect = Rectangle((fusion_x-1, fusion_y-1), 2, 2, 
                    facecolor=colors['fusion'], edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(fusion_x, fusion_y, 'Temporal\nFusion\nNetwork', 
           ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Output Classification
    output_x = 18
    output_y = 6.5
    rect = Rectangle((output_x-0.6, output_y-0.5), 1.2, 1, 
                    facecolor=colors['output'], edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(output_x, output_y, 'Change\nClassification\n(Binary)', 
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows - Input to VGG16
    for i in range(3):
        x_start = 1 + i * 2
        x_end = 8 + i * 2
        # Before to VGG16
        ax.annotate('', xy=(x_end-0.6, vgg_y+0.4), xytext=(x_start+0.4, before_y),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
        # After to VGG16
        ax.annotate('', xy=(x_end-0.6, vgg_y-0.4), xytext=(x_start+0.4, after_y),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    
    # Arrows - VGG16 to Siamese
    for i in range(3):
        x = 8 + i * 2
        ax.annotate('', xy=(x, siamese_y+0.5), xytext=(x, vgg_y-0.8),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='blue'))
    
    # Arrows - Siamese to Fusion
    for i in range(3):
        x_start = 8 + i * 2
        ax.annotate('', xy=(fusion_x-1, fusion_y), xytext=(x_start+0.6, siamese_y),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='orange'))
    
    # Arrow - Fusion to Output
    ax.annotate('', xy=(output_x-0.6, output_y), xytext=(fusion_x+1, fusion_y),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=colors['input'], label='Input Images'),
        mpatches.Patch(color=colors['vgg'], label='VGG16 Feature Extractor'),
        mpatches.Patch(color=colors['siamese'], label='Siamese Network'),
        mpatches.Patch(color=colors['fusion'], label='Temporal Fusion'),
        mpatches.Patch(color=colors['output'], label='Output Classification')
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=5, 
             bbox_to_anchor=(0.5, 0.02), frameon=True, fancybox=True, shadow=True)
    
    # Add feature dimensions as annotations
    ax.text(10, 3, 'Feature Dimensions:\nVGG16: 512-D → Siamese: 256-D → Fusion: 128-D → Output: 2-D', 
           ha='center', va='center', fontsize=11, 
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"Architecture diagram saved to: {save_path}")
    return fig

def create_confusion_matrix_plot(y_true, y_pred, save_path='/content/confusion_matrix.png'):
    """
    Create publication-quality confusion matrix
    Suitable for: Results section
    """
    from sklearn.metrics import confusion_matrix, classification_report
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Confusion Matrix Heatmap
    class_names = ['No Change', 'Change Detected']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Number of Samples'})
    ax1.set_title('(a) Confusion Matrix')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Plot 2: Performance Metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    values = [accuracy, precision, recall, f1, specificity]
    
    bars = ax2.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_title('(b) Performance Metrics')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-axis labels for better readability
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed metrics
    print("="*50)
    print("CLASSIFICATION METRICS")
    print("="*50)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print("="*50)
    
    print(f"Confusion matrix plot saved to: {save_path}")
    return fig

def create_dataset_analysis_plot(data_directory, save_path='/content/dataset_analysis.png'):
    """
    Create dataset analysis visualization
    Suitable for: Dataset section
    """
    import os
    
    # Analyze dataset
    samples = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
    
    change_count = 0
    no_change_count = 0
    modality_counts = {'rgb': 0, 'thermal': 0, 'lidar': 0}
    
    for sample in samples:
        sample_path = os.path.join(data_directory, sample)
        
        # Check label
        label_path = os.path.join(sample_path, 'label.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = int(f.read().strip())
                if label == 1:
                    change_count += 1
                else:
                    no_change_count += 1
        
        # Check modalities
        for time_period in ['before', 'after']:
            period_path = os.path.join(sample_path, time_period)
            if os.path.exists(period_path):
                for modality in ['rgb', 'thermal', 'lidar']:
                    file_path = os.path.join(period_path, f'{modality}.png')
                    if os.path.exists(file_path):
                        modality_counts[modality] += 1
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Class Distribution
    labels = ['No Change', 'Change Detected']
    sizes = [no_change_count, change_count]
    colors = ['#ff9999', '#66b3ff']
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                      startangle=90, textprops={'fontsize': 11})
    ax1.set_title('(a) Class Distribution')
    
    # Plot 2: Sample Count Bar Chart
    ax2.bar(labels, sizes, color=colors, edgecolor='black', linewidth=1)
    ax2.set_title('(b) Sample Counts')
    ax2.set_ylabel('Number of Samples')
    
    # Add count labels on bars
    for i, v in enumerate(sizes):
        ax2.text(i, v + max(sizes)*0.01, str(v), ha='center', va='bottom', fontweight='bold')
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Modality Availability
    modalities = list(modality_counts.keys())
    counts = list(modality_counts.values())
    
    bars = ax3.bar(modalities, counts, color=['#ff7f0e', '#2ca02c', '#d62728'], 
                   edgecolor='black', linewidth=1)
    ax3.set_title('(c) Modality Availability')
    ax3.set_ylabel('Number of Images')
    ax3.set_xlabel('Modality Type')
    
    # Add count labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Dataset Statistics Summary
    ax4.axis('off')
    
    total_samples = len(samples)
    balance_ratio = change_count / total_samples if total_samples > 0 else 0
    
    stats_text = f"""
    DATASET STATISTICS
    ==================
    
    Total Samples: {total_samples}
    
    Class Distribution:
    • No Change: {no_change_count} ({no_change_count/total_samples*100:.1f}%)
    • Change: {change_count} ({change_count/total_samples*100:.1f}%)
    
    Class Balance Ratio: {balance_ratio:.3f}
    
    Modality Coverage:
    • RGB: {modality_counts['rgb']} images
    • Thermal: {modality_counts['thermal']} images  
    • LiDAR: {modality_counts['lidar']} images
    
    Dataset Quality: {'Balanced' if 0.3 <= balance_ratio <= 0.7 else 'Imbalanced'}
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Dataset analysis plot saved to: {save_path}")
    return fig

def create_complete_publication_figure_set(results, data_directory, 
                                         y_true=None, y_pred=None,
                                         base_path='/content/publication_figures'):
    """
    Create complete set of publication-quality figures
    """
    import os
    os.makedirs(base_path, exist_ok=True)
    
    print("Creating complete publication figure set...")
    print("="*60)
    
    # Figure 1: Architecture Diagram
    print("1. Creating architecture diagram...")
    create_architecture_diagram(f'{base_path}/Fig1_Architecture.png')
    
    # Figure 2: Dataset Analysis
    print("2. Creating dataset analysis...")
    create_dataset_analysis_plot(data_directory, f'{base_path}/Fig2_Dataset_Analysis.png')
    
    # Figure 3: Training Curves
    print("3. Creating training curves...")
    create_training_curves_figure(results, f'{base_path}/Fig3_Training_Curves.png')
    
    # Figure 4: Confusion Matrix (if evaluation data provided)
    if y_true is not None and y_pred is not None:
        print("4. Creating confusion matrix...")
        create_confusion_matrix_plot(y_true, y_pred, f'{base_path}/Fig4_Confusion_Matrix.png')
    else:
        print("4. Skipping confusion matrix (no evaluation data provided)")
    
    print("="*60)
    print(f"✅ All publication figures saved to: {base_path}/")
    print("Figures are ready for journal submission!")
    
    return base_path

# Usage instructions
def print_usage_instructions():
    """Print instructions for using publication plots"""
    print("="*60)
    print("PUBLICATION PLOTS USAGE GUIDE")
    print("="*60)
    print("""
    After training with visualization, use these functions:
    
    1. For training curves:
       create_training_curves_figure(results, 'training_curves.png')
    
    2. For architecture diagram:
       create_architecture_diagram('architecture.png')
    
    3. For dataset analysis:
       create_dataset_analysis_plot(data_directory, 'dataset.png')
    
    4. For confusion matrix (after evaluation):
       create_confusion_matrix_plot(y_true, y_pred, 'confusion.png')
    
    5. For complete figure set:
       create_complete_publication_figure_set(results, data_directory)
    
    All figures are saved at 300 DPI with publication-quality formatting.
    """)
    print("="*60)