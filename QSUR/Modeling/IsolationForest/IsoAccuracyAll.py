import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import recall_score

def calculate_recall(true_labels: np.ndarray, scores: np.ndarray, target_usecase: str) -> tuple:
    """Calculate recall values for different thresholds."""
    # Convert true labels to binary
    y_true = (true_labels == target_usecase).astype(int)
    n_positives = sum(y_true)
    
    if n_positives < 2 or np.std(scores) == 0:
        print(f"Skipping {target_usecase} - insufficient data or no score variation")
        return None, None
    
    # Generate thresholds specific to this model
    model_thresholds = np.linspace(scores.min(), scores.max(), 100)
    recall_values = []
    
    for threshold in model_thresholds:
        predictions = (scores <= threshold).astype(int)
        recall = recall_score(y_true, predictions, zero_division=0)
        recall_values.append(recall)
    
    return np.array(recall_values), model_thresholds

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import recall_score

def calculate_recall(true_labels: np.ndarray, scores: np.ndarray, target_usecase: str) -> tuple:
    """Calculate recall values for different thresholds."""
    # Convert true labels to binary
    y_true = (true_labels == target_usecase).astype(int)
    n_positives = sum(y_true)
    
    # Add debug prints
    print(f"Processing {target_usecase}")
    print(f"Number of positive samples: {n_positives}")
    print(f"Score range: {scores.min():.3f} to {scores.max():.3f}")
    
    if n_positives < 2 or np.std(scores) == 0:
        print(f"Skipping {target_usecase} - insufficient data or no score variation")
        return None, None
    
    # Generate thresholds specific to this model
    model_thresholds = np.linspace(scores.min(), scores.max(), 100)
    recall_values = []
    
    # Changed comparison direction for anomaly scores
    for threshold in model_thresholds:
        predictions = (scores >= threshold).astype(int)  # Changed to >= from <=
        recall = recall_score(y_true, predictions, zero_division=0)
        recall_values.append(recall)
    
    return np.array(recall_values), model_thresholds

def analyze_scores_for_paper(true_labels_path: str, scores_path: str, output_dir: str):
    """Generate recall plots for paper."""
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
        
        # Calculate recall for this model
        recall_values, model_thresholds = calculate_recall(
            true_use_cases, 
            scores, 
            use_case
        )
        
        if recall_values is not None:
            # Store valid results
            all_results.append((recall_values, model_thresholds))
            model_names.append(use_case)
            
            # Plot individual model recall curve with more informative labels
            plt.figure(figsize=(10, 6))
            plt.plot(model_thresholds, recall_values, 'b-', linewidth=2, label='Recall')
            plt.xlabel('Anomaly Score Threshold')
            plt.ylabel('Recall')
            plt.title(f'Model Performance: {use_case}\n(n_samples={len(scores)})')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add score distribution
            ax2 = plt.gca().twinx()
            ax2.hist(scores, bins=50, alpha=0.3, color='gray')
            ax2.set_ylabel('Score Distribution')
            
            plt.savefig(output_dir / f'{model}_recall.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Plot mean recall across all models
    if all_results:
        print(f"Generating summary plots for {len(all_results)} valid models")
        common_thresholds = np.linspace(-0.3, 0.3, 100)
        interpolated_recalls = []
        
        for recall_values, model_thresholds in all_results:
            interp_recall = np.interp(common_thresholds, model_thresholds, recall_values)
            interpolated_recalls.append(interp_recall)
        
        mean_recall = np.mean(interpolated_recalls, axis=0)
        std_recall = np.std(interpolated_recalls, axis=0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(common_thresholds, mean_recall, 'b-', linewidth=2, label='Mean Recall')
        plt.fill_between(common_thresholds, 
                        mean_recall - std_recall, 
                        mean_recall + std_recall, 
                        alpha=0.2, color='b', label='±1 std dev')
        plt.xlabel('Anomaly Score Threshold')
        plt.ylabel('Recall')
        plt.title(f'Average Model Performance\n(across {len(all_results)} models)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(output_dir / 'mean_recall_with_std.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Enhanced heatmap
        key_thresholds = [-0.2, -0.1, 0, 0.1, 0.2]
        recall_matrix = []
        
        for recall in interpolated_recalls:
            recalls_at_thresh = [recall[np.abs(common_thresholds - t).argmin()] 
                               for t in key_thresholds]
            recall_matrix.append(recalls_at_thresh)
        
        plt.figure(figsize=(12, max(6, len(model_names)/2)))
        im = plt.imshow(recall_matrix, aspect='auto', cmap='YlOrRd')
        plt.colorbar(im, label='Recall')
        plt.xticks(range(len(key_thresholds)), [f'{t:.2f}' for t in key_thresholds])
        plt.yticks(range(len(model_names)), model_names)
        plt.xlabel('Threshold')
        plt.ylabel('Use Case')
        plt.title(f'Recall Values at Different Thresholds\n(across {len(model_names)} models)')
        plt.tight_layout()
        plt.savefig(output_dir / 'recall_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("No valid results to plot")

if __name__ == "__main__":
    analyze_scores_for_paper(
        true_labels_path="../../../Data/Curated/UseCaseDataModeling.csv",
        scores_path="../../../Results/IsoForest/TrainResults/anomaly_scores.csv",
        output_dir="../../../Results/IsoForest/Results/precision_recall_plots"
    )