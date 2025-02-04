import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler
import joblib
from glob import glob

def load_model_and_data(base_path: Path, class_num: int):
    """Load BGMM model and all datasets, separating target class from others."""
    try:
        model = joblib.load(base_path / f"bgmm_{class_num}.pkl")
        target_data = pd.read_csv(base_path / f"umap_{class_num}.csv")
        
        other_data = []
        for data_file in base_path.glob("umap_*.csv"):
            other_class_num = int(data_file.stem.split('_')[1])
            if other_class_num != class_num:
                other_data.append(pd.read_csv(data_file))
        
        other_data = pd.concat(other_data, ignore_index=True)
        return model, target_data, other_data
    
    except Exception as e:
        print(f"Error loading model or data for class {class_num}: {e}")
        return None, None, None

def get_normalized_scores(model, target_data: pd.DataFrame, other_data: pd.DataFrame) -> tuple:
    """Calculate and normalize likelihood scores."""
    # Get log likelihood scores
    target_scores = model.score_samples(target_data)
    other_scores = model.score_samples(other_data)
    
    # Print raw score statistics
    print("\nRaw score statistics:")
    print(f"Target scores: min={target_scores.min():.3f}, max={target_scores.max():.3f}")
    print(f"Other scores: min={other_scores.min():.3f}, max={other_scores.max():.3f}")
    
    # Combine scores for normalization
    all_scores = np.concatenate([target_scores, other_scores])
    
    # Remove extreme outliers (values below 1st percentile or above 99th percentile)
    p1, p99 = np.percentile(all_scores, [1, 99])
    all_scores = np.clip(all_scores, p1, p99)
    
    # Normalize scores to [0, 1] range
    scaler = MinMaxScaler()
    normalized_scores = scaler.fit_transform(all_scores.reshape(-1, 1)).ravel()
    
    # Split back into target and other scores
    n_target = len(target_scores)
    normalized_target_scores = normalized_scores[:n_target]
    normalized_other_scores = normalized_scores[n_target:]
    
    # Print normalized score statistics
    print("\nNormalized score statistics:")
    print(f"Target scores: min={normalized_target_scores.min():.3f}, max={normalized_target_scores.max():.3f}")
    print(f"Other scores: min={normalized_other_scores.min():.3f}, max={normalized_other_scores.max():.3f}")
    
    return normalized_target_scores, normalized_other_scores

def calculate_metrics(model, target_data: pd.DataFrame, other_data: pd.DataFrame) -> tuple:
    """Calculate precision and recall values for different threshold levels."""
    # Get normalized scores
    target_scores, other_scores = get_normalized_scores(model, target_data, other_data)
    
    # Combine scores and create true labels
    all_scores = np.concatenate([target_scores, other_scores])
    true_labels = np.concatenate([np.ones(len(target_scores)), np.zeros(len(other_scores))])
    
    print(f"\nNumber of positive samples: {len(target_scores)}")
    print(f"Number of negative samples: {len(other_scores)}")
    print(f"Target scores stats: mean={target_scores.mean():.3f}, std={target_scores.std():.3f}")
    print(f"Other scores stats: mean={other_scores.mean():.3f}, std={other_scores.std():.3f}")
    
    # Generate thresholds from 0 to 1
    thresholds = np.linspace(0, 1, 100)
    recall_values = []
    precision_values = []
    
    for threshold in thresholds:
        predictions = (all_scores >= threshold).astype(int)
        recall = recall_score(true_labels, predictions, zero_division=0)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall_values.append(recall)
        precision_values.append(precision)
    
    return np.array(recall_values), np.array(precision_values), thresholds
def analyze_bgmm_models(base_path: str, output_dir: str):
    """Analyze each BGMM model's ability to identify its own class."""
    base_path = Path(base_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    class_numbers = []
    
    # Find all BGMM models
    model_files = sorted(glob(str(base_path / "bgmm_*.pkl")))
    
    for model_file in model_files:
        class_num = int(Path(model_file).stem.split("_")[1])
        print(f"\nProcessing model for class {class_num}")
        
        # Load model and data
        model, target_data, other_data = load_model_and_data(base_path, class_num)
        if model is None:
            continue
        
        # Calculate metrics
        recall_values, precision_values, thresholds = calculate_metrics(
            model, target_data, other_data
        )
        
        all_results.append((recall_values, precision_values, thresholds))
        class_numbers.append(class_num)
        
        # Plot individual model metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Metrics vs threshold
        ax1.plot(thresholds, recall_values, 'b-', linewidth=2, label='Recall')
        ax1.plot(thresholds, precision_values, 'r-', linewidth=2, label='Precision')
        ax1.set_xlabel('Log-Likelihood Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title(f'Metrics vs Threshold: Class {class_num}\n' + 
                     f'(positives={len(target_data)}, negatives={len(other_data)})')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Precision-recall curve
        ax2.plot(recall_values, precision_values, 'g-', linewidth=2)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.grid(True, alpha=0.3)
        
        # Add threshold markers
        percentile_points = [10, 30, 50, 70, 90]
        for p in percentile_points:
            threshold = np.percentile(thresholds, p)
            idx = np.abs(thresholds - threshold).argmin()
            ax2.plot(recall_values[idx], precision_values[idx], 'ko', markersize=5)
            ax2.annotate(f'{threshold:.1f}', 
                        (recall_values[idx], precision_values[idx]),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'class_{class_num}_precision_recall.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot mean metrics across all models
    if all_results:
        print(f"\nGenerating summary plots for {len(all_results)} models")
        
        # Find global score range for proper interpolation
        all_thresholds = np.concatenate([t for _, _, t in all_results])
        global_range = np.percentile(all_thresholds, [1, 99])
        common_thresholds = np.linspace(global_range[0], global_range[1], 100)
        
        interpolated_recalls = []
        interpolated_precisions = []
        
        for recall_values, precision_values, thresholds in all_results:
            # Interpolate to common threshold grid
            interp_recall = np.interp(common_thresholds, thresholds, recall_values)
            interp_precision = np.interp(common_thresholds, thresholds, precision_values)
            interpolated_recalls.append(interp_recall)
            interpolated_precisions.append(interp_precision)
        
        mean_recall = np.mean(interpolated_recalls, axis=0)
        std_recall = np.std(interpolated_recalls, axis=0)
        mean_precision = np.mean(interpolated_precisions, axis=0)
        std_precision = np.std(interpolated_precisions, axis=0)
        
        # Create summary plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Average metrics vs threshold
        ax1.plot(common_thresholds, mean_recall, 'b-', linewidth=2, label='Mean Recall')
        ax1.fill_between(common_thresholds, 
                        mean_recall - std_recall, 
                        mean_recall + std_recall, 
                        alpha=0.2, color='b')
        
        ax1.plot(common_thresholds, mean_precision, 'r-', linewidth=2, label='Mean Precision')
        ax1.fill_between(common_thresholds, 
                        mean_precision - std_precision, 
                        mean_precision + std_precision, 
                        alpha=0.2, color='r')
        
        ax1.set_xlabel('Log-Likelihood Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title(f'Average Metrics vs Threshold\n(across {len(all_results)} models)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Average precision-recall curve
        ax2.plot(mean_recall, mean_precision, 'g-', linewidth=2)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Average Precision-Recall Curve')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'mean_precision_recall.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create heatmaps
        key_percentiles = [10, 30, 50, 70, 90]
        precision_matrix = []
        recall_matrix = []
        
        for recall_values, precision_values, thresholds in all_results:
            # Get values at specific threshold percentiles
            threshold_points = np.percentile(thresholds, key_percentiles)
            recalls_at_thresh = [recall_values[np.abs(thresholds - t).argmin()] 
                               for t in threshold_points]
            precisions_at_thresh = [precision_values[np.abs(thresholds - t).argmin()] 
                                  for t in threshold_points]
            recall_matrix.append(recalls_at_thresh)
            precision_matrix.append(precisions_at_thresh)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, max(6, len(class_numbers)/2)))
        
        # Recall heatmap
        im1 = ax1.imshow(recall_matrix, aspect='auto', cmap='YlOrRd')
        plt.colorbar(im1, ax=ax1, label='Recall')
        ax1.set_xticks(range(len(key_percentiles)))
        ax1.set_xticklabels([f'{p}th' for p in key_percentiles])
        ax1.set_yticks(range(len(class_numbers)))
        ax1.set_yticklabels([f'Class {c}' for c in class_numbers])
        ax1.set_xlabel('Score Percentile')
        ax1.set_ylabel('Class')
        ax1.set_title('Recall Values at Different Score Percentiles')
        
        # Precision heatmap
        im2 = ax2.imshow(precision_matrix, aspect='auto', cmap='YlOrRd')
        plt.colorbar(im2, ax=ax2, label='Precision')
        ax2.set_xticks(range(len(key_percentiles)))
        ax2.set_xticklabels([f'{p}th' for p in key_percentiles])
        ax2.set_yticks(range(len(class_numbers)))
        ax2.set_yticklabels([f'Class {c}' for c in class_numbers])
        ax2.set_xlabel('Score Percentile')
        ax2.set_title('Precision Values at Different Score Percentiles')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("No valid results to plot")

if __name__ == "__main__":
    analyze_bgmm_models(
        base_path="/home/matt/Proj/QSURv3/Data/ReducedForBGMM",
        output_dir="/home/matt/Proj/QSURv3/Results/BGMM/precision_recall_plots"
    )