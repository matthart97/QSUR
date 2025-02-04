import pandas as pd
import numpy as np
from pathlib import Path
from TrainBGMM import UMAPBGMMPipeline

def run_inference(data_path, models_dir, output_dir='inference_results'):
    """
    Run inference on a dataset to generate prediction profiles and most likely use cases.
    
    Args:
        data_path (str): Path to the input data CSV
        models_dir (str): Directory containing saved models
        output_dir (str): Directory to save results
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data and models
    print("Loading data and models...")
    data = pd.read_csv(data_path)
    pipeline = UMAPBGMMPipeline.load_models(models_dir)
    
    # Initialize results storage
    all_probabilities = {}
    use_case_predictions = []
    molecule_ids = data.index.tolist()
    use_cases = sorted(pipeline.models.keys())
    
    print("Running predictions...")
    # For each molecule, get predictions for all use cases
    for mol_idx in molecule_ids:
        mol_data = data.iloc[[mol_idx]]
        mol_probs = {}
        
        # Get predictions for each use case
        for use_case in use_cases:
            try:
                reduced_data, _ = pipeline.transform(mol_data, use_case)
                component_probs = pipeline.models[use_case]['bgmm'].predict_proba(reduced_data)[0]
                mol_probs[use_case] = np.max(component_probs)
            except Exception as e:
                print(f"Error processing molecule {mol_idx} for use case {use_case}: {str(e)}")
                mol_probs[use_case] = 0.0
        
        # Store probabilities and find most likely use case
        all_probabilities[mol_idx] = mol_probs
        most_likely_use = max(mol_probs.items(), key=lambda x: x[1])
        use_case_predictions.append({
            'molecule_id': mol_idx,
            'predicted_use': most_likely_use[0],
            'confidence': most_likely_use[1]
        })
    
    # Create DataFrames
    prob_df = pd.DataFrame.from_dict(all_probabilities, orient='index')
    prob_df.columns = [f'use_case_{col}' for col in prob_df.columns]
    predictions_df = pd.DataFrame(use_case_predictions)
    
    # Save results
    prob_df.to_csv(Path(output_dir) / 'probability_profiles.csv')
    predictions_df.to_csv(Path(output_dir) / 'predicted_uses.csv', index=False)
    
    # Print summary
    print("\nPrediction Summary:")
    print(f"Total molecules processed: {len(molecule_ids)}")
    print("\nUse case distribution:")
    use_dist = predictions_df['predicted_use'].value_counts()
    for use, count in use_dist.items():
        print(f"Use case {use}: {count} molecules ({count/len(molecule_ids)*100:.1f}%)")
    
    return prob_df, predictions_df

# Example usage
if __name__ == "__main__":
    # Just update these paths to run inference
    data_path = "/home/matt/Proj/QSURv3/PatentMolProcessing/intermediate_results_10.csv"
    models_dir = "/home/matt/Proj/QSURv3/QSUR/Models/BGMMs"
    output_dir = "/home/matt/Proj/QSURv3/Results/BGMM"
    
    probability_profiles, predictions = run_inference(data_path, models_dir, output_dir)