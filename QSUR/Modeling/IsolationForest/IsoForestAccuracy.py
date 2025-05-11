import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import recall_score, precision_score

def calculate_metrics(true_labels: np.ndarray, scores: np.ndarray, target_usecase: str) -> tuple:
    """Calculate precision and recall values for different thresholds."""
    # Convert true labels to binary
    y_true = (true_labels == target_usecase).astype(int)
    n_positives = sum(y_true)
    
    # Add debug prints
    print(f"Processing {target_usecase}")
    print(f"Number of positive samples: {n_positives}")
    print(f"Score range: {scores.min():.3f} to {scores.max():.3f}")
    
    if n_positives < 2 or np.std(scores) == 0:
        print(f"Skipping {target_usecase} - insufficient data or no score variation")
        return None, None, None
    
    # Generate thresholds specific to this model
    model_thresholds = np.linspace(scores.min(), scores.max(), 100)
    recall_values = []
    precision_values = []
    
    for threshold in model_thresholds:
        predictions = (scores >= threshold).astype(int)
        recall = recall_score(y_true, predictions, zero_division=0)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall_values.append(recall)
        precision_values.append(precision)
    
    return np.array(recall_values), np.array(precision_values), model_thresholds

def analyze_scores_for_paper(true_labels_path: str, scores_path: str, output_dir: str):
    """Generate precision-recall plots for paper."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data with validation
    try:
        true_labels = pd.read_csv(true_labels_path, low_memory=False)
        scores_df = pd.read_csv(scores_path)
        
        print(f"Loaded {len(true_labels)} true labels and {len(scores_df)} scores")
        
        if len(true_labels) == 0 or len(scores_df) == 0:
            raise ValueError("Empty datasets loaded")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Store results
    all_results = []
    model_names = []
    
    # Process each model
    for model in scores_df.columns:
        use_case = model.replace('_', '/')
        scores = scores_df[model].values
        
        # Add validation for NaN values
        if np.isnan(scores).any():
            print(f"Warning: Found NaN values in scores for {model}")
            scores = np.nan_to_num(scores, nan=np.nanmean(scores))
        
        true_use_cases = true_labels['Harmonized Functional Use'].values
        
        # Calculate metrics for this model
        recall_values, precision_values, model_thresholds = calculate_metrics(
            true_use_cases, 
            scores, 
            use_case
        )
        
        if recall_values is not None:
            # Store valid results
            all_results.append((recall_values, precision_values, model_thresholds))
            model_names.append(use_case)
            
            # Plot individual model precision-recall curve
            plt.figure(figsize=(12, 6))
            
            # Create two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot metrics vs threshold
            ax1.plot(model_thresholds, recall_values, 'b-', linewidth=2, label='Recall')
            ax1.plot(model_thresholds, precision_values, 'r-', linewidth=2, label='Precision')
            ax1.set_xlabel('Anomaly Score Threshold')
            ax1.set_ylabel('Score')
            ax1.set_title(f'Metrics vs Threshold: {use_case}\n(n_samples={len(scores)})')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot precision-recall curve
            ax2.plot(recall_values, precision_values, 'g-', linewidth=2)
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title('Precision-Recall Curve')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{model}_precision_recall.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Plot mean metrics across all models
    if all_results:
        print(f"Generating summary plots for {len(all_results)} valid models")
        common_thresholds = np.linspace(-0.3, 0.3, 100)
        interpolated_recalls = []
        interpolated_precisions = []
        
        for recall_values, precision_values, model_thresholds in all_results:
            interp_recall = np.interp(common_thresholds, model_thresholds, recall_values)
            interp_precision = np.interp(common_thresholds, model_thresholds, precision_values)
            interpolated_recalls.append(interp_recall)
            interpolated_precisions.append(interp_precision)
        
        mean_recall = np.mean(interpolated_recalls, axis=0)
        std_recall = np.std(interpolated_recalls, axis=0)
        mean_precision = np.mean(interpolated_precisions, axis=0)
        std_precision = np.std(interpolated_precisions, axis=0)
        
        # Plot average metrics vs threshold
        plt.figure(figsize=(12, 6))
        
        # Create two subplots for the summary
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Metrics vs Threshold
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
        
        ax1.set_xlabel('Anomaly Score Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title(f'Average Metrics vs Threshold\n(across {len(all_results)} models)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Average Precision-Recall Curve
        ax2.plot(mean_recall, mean_precision, 'g-', linewidth=2)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Average Precision-Recall Curve')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'mean_precision_recall.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Enhanced heatmap for both metrics
        key_thresholds = [-0.2, -0.1, 0, 0.1, 0.2]
        precision_matrix = []
        recall_matrix = []
        
        for recall, precision, _ in all_results:
            recalls_at_thresh = [recall[np.abs(common_thresholds - t).argmin()] 
                               for t in key_thresholds]
            precisions_at_thresh = [precision[np.abs(common_thresholds - t).argmin()] 
                                  for t in key_thresholds]
            recall_matrix.append(recalls_at_thresh)
            precision_matrix.append(precisions_at_thresh)
        
        # Create subplots for precision and recall heatmaps
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, max(6, len(model_names)/2)))
        
        # Recall heatmap
        im1 = ax1.imshow(recall_matrix, aspect='auto', cmap='YlOrRd')
        plt.colorbar(im1, ax=ax1, label='Recall')
        ax1.set_xticks(range(len(key_thresholds)))
        ax1.set_xticklabels([f'{t:.2f}' for t in key_thresholds])
        ax1.set_yticks(range(len(model_names)))
        ax1.set_yticklabels(model_names)
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Use Case')
        ax1.set_title('Recall Values at Different Thresholds')
        
        # Precision heatmap
        im2 = ax2.imshow(precision_matrix, aspect='auto', cmap='YlOrRd')
        plt.colorbar(im2, ax=ax2, label='Precision')
        ax2.set_xticks(range(len(key_thresholds)))
        ax2.set_xticklabels([f'{t:.2f}' for t in key_thresholds])
        ax2.set_yticks(range(len(model_names)))
        ax2.set_yticklabels(model_names)
        ax2.set_xlabel('Threshold')
        ax2.set_title('Precision Values at Different Thresholds')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("No valid results to plot")

if __name__ == "__main__":
    analyze_scores_for_paper(
        true_labels_path="../../../Data/Curated/UseCaseDataModeling.csv",
        scores_path="../../../Results/IsoForest/TrainResults/anomaly_scores.csv",
        output_dir="../../../Results/IsoForest/Results/precision_recall_plots"
    )