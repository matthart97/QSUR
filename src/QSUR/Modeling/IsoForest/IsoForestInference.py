from pathlib import Path
import pandas as pd
import numpy as np
from joblib import load
from typing import Dict, List, Union, Optional

class IsolationForestInference:
    def __init__(self, models_dir: str, feature_prefix: str = 'Bit'):
        """
        Initialize the inference class.
        
        Args:
            models_dir: Directory containing saved models
            feature_prefix: Prefix for feature columns
        """
        self.models_dir = Path(models_dir)
        self.feature_prefix = feature_prefix
        self.models = {}
        self.load_all_models()
        
    def load_all_models(self) -> None:
        """Load all saved models from the models directory."""
        print(f"Looking for models in: {self.models_dir}")
        model_files = list(self.models_dir.glob('*.joblib'))
        print(f"Found model files: {model_files}")
        
        for model_path in model_files:
            model_name = model_path.stem
            model_data = load(model_path)
            self.models[model_name] = model_data
            
    def run_inference(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Run inference and get anomaly scores for input data using all loaded models.
        
        Args:
            data: DataFrame containing features with same structure as training data
            
        Returns:
            Dictionary mapping model names to anomaly scores
        """
        results = {}
        
        for model_name, model_data in self.models.items():
            # Get feature columns in correct order
            features = model_data['feature_columns']
            X = data[features]
            
            # Get anomaly scores (higher scores mean more normal, lower scores mean more anomalous)
            scores = model_data['model'].decision_function(X)
            
            # Store results
            results[model_name] = scores
            
        return results

def run_batch_inference(data_path: str, models_dir: str, output_path: Optional[str] = None, feature_prefix: str = 'Bit'):
    # Load data
    data = pd.read_csv(data_path, low_memory=False)  # Added low_memory=False to handle mixed types
    print(f"Loaded data with shape: {data.shape}")
    
    # Initialize inference - FIXED HERE
    inferencer = IsolationForestInference(models_dir, feature_prefix)
    print(f"Number of models loaded: {len(inferencer.models)}")
    
    # Run inference
    results = inferencer.run_inference(data)
    
    # Save results
    if output_path:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
    
    return results
if __name__ == "__main__":
    # Example usage
    results = run_batch_inference(
        data_path='/home/matt/Proj/QSURv3/Data/Curated/UseCaseDataModeling.csv',
        models_dir='/home/matt/Proj/QSURv3/QSUR/Models/IsoForestModels/models/',
        output_path='/home/matt/Proj/QSURv3/Results/IsoForest/TrainResults/anomaly_scores.csv'
    )