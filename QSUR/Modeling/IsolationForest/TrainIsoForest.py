"""
This code is for implementing the training of the Isolation Forst
"""
from IsoForestTrainer import IsolationForestTrainer
import pandas as pd
from pathlib import Path
from typing import Dict

class TrainModels:
    def __init__(self, 
                 data_path: str,
                 output_dir: str,
                 class_column: str = 'Harmonized Functional Use',
                 n_estimators: int = 100,
                 contamination: float = 'auto',
                 feature_prefix: str = 'Bit'):
        """
        Initialize the trainer.
        
        Args:
            data_path: Path to the input CSV file
            output_dir: Directory to save models
            class_column: Name of the class column
            n_estimators: Number of estimators for IsolationForest
            contamination: Expected proportion of outliers
            feature_prefix: Prefix for feature columns
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.class_column = class_column
        self.model_params = {
            'n_estimators': n_estimators,
            'contamination': contamination,
            'feature_prefix': feature_prefix
        }
        
        # Create output directory
        self.models_dir = self.output_dir / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def train_all_models(self) -> Dict[str, IsolationForestTrainer]:
        """
        Train Isolation Forest models for all use cases in the dataset.
        
        Returns:
            Dictionary mapping use case names to trained models
        """
        # Load data
        data = pd.read_csv(self.data_path)
        cases = data[self.class_column].unique()
        
        # Store trained models
        trained_models = {}
        
        # Train models for each use case
        for use_case in cases:
            try:
                # Create safe filename
                safe_name = str(use_case).replace('/', '_')
                print(f"Training model for {safe_name}")
                
                # Initialize and train model
                trainer = IsolationForestTrainer(**self.model_params)
                X, _ = trainer.prepare_data(data, self.class_column, use_case)
                trainer.train(X)
                
                # Save model
                model_path = self.models_dir / f'{safe_name}.joblib'
                trainer.save_model(model_path)
                
                # Store trained model
                trained_models[safe_name] = trainer
                
                print(f"Successfully trained and saved model for {safe_name}")
                
            except Exception as e:
                print(f"Error processing {use_case}: {str(e)}")
                continue
        
        return trained_models

def run_training(data_path: str, 
                output_dir: str,
                class_column: str = 'Harmonized Functional Use',
                **model_params) -> Dict[str, IsolationForestTrainer]:
    """
    Train models for all use cases.
    
    Args:
        data_path: Path to input data
        output_dir: Path to save models
        class_column: Name of the class column
        **model_params: Parameters for the IsolationForest model
    
    Returns:
        Dictionary of trained models
    """
    trainer = TrainModels(
        data_path=data_path,
        output_dir=output_dir,
        class_column=class_column,
        **model_params
    )
    
    return trainer.train_all_models()

if __name__ == "__main__":
    # Example usage
    models = run_training(
        data_path='../../../Data/Curated/UseCaseDataModeling.csv',
        output_dir='../../../QSUR/Models/IsoForestModels',
        n_estimators=150,
        contamination=0.1
    )