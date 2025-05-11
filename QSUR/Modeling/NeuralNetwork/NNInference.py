import torch
import pandas as pd
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import numpy as np
from TrainNN import MoleculeNet, Config

class MoleculePredictor:
    def __init__(self, model_dir, label_mapping_path):
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load label mapping
        with open(label_mapping_path, 'r') as f:
            self.label_mapping = json.load(f)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize model
        self.model = self._load_model()
        self.scaler = StandardScaler()
        
        print(f"Model loaded successfully from {model_dir}")
        print(f"Label mapping loaded from {label_mapping_path}")
    
    def _load_config(self):
        config_path = self.model_dir / 'config.json'
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return Config(**config_dict)
    
    def _load_model(self):
        # Find the best checkpoint
        checkpoint_path = max(self.model_dir.glob('models/checkpoint_*.pt'),
                            key=lambda p: float(p.stem.split('_')[2]))
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get model output size from checkpoint
        output_size = checkpoint['model_state_dict']['model.9.weight'].shape[0]
        
        # Initialize and load the model
        model = MoleculeNet(
            config=self.config,
            input_dim=self._get_input_dim(),
            num_classes=output_size
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def _get_input_dim(self):
        # Read the first row of your training data to get input dimension
        df = pd.read_csv(self.config.data_path, low_memory=False)
        return len(df.filter(regex=f'^{self.config.feature_prefix}').columns)
    
    def predict(self, input_features, smiles_column='SMILES'):
        """
        Make predictions for input features and save simplified results
        
        Args:
            input_features: DataFrame with feature columns and SMILES
            smiles_column: Name of the SMILES column in the DataFrame
        
        Returns:
            Dictionary with predicted class and probabilities
        """
        # Ensure SMILES column exists
        if smiles_column not in input_features.columns:
            raise ValueError(f"SMILES column '{smiles_column}' not found in input data")
        
        # Ensure input features match the expected format
        features = input_features.filter(regex=f'^{self.config.feature_prefix}')
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(scaled_features).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        # Convert predictions to numpy
        probabilities = probabilities.cpu().numpy()
        predicted_class = predicted_class.cpu().numpy()
        
        # Map predicted class indices to use cases
        predicted_uses = [self.get_use_case(idx) for idx in predicted_class]
        
        # Create results
        results = []
        for i in range(len(predicted_class)):
            result = {
                'predicted_class_index': int(predicted_class[i]),
                'predicted_use': predicted_uses[i],
                'confidence': float(probabilities[i][predicted_class[i]]),
                'probabilities': {
                    idx: float(prob) for idx, prob in enumerate(probabilities[i])
                }
            }
            results.append(result)
        
        # Create simplified results DataFrame
        simplified_results = pd.DataFrame({
            'SMILES': input_features[smiles_column],
            'Predicted_Class_Index': [r['predicted_class_index'] for r in results],
            'Predicted_Use': [r['predicted_use'] for r in results],
            'Confidence': [r['confidence'] for r in results]
        })
        
        return results, simplified_results
    
    def get_use_case(self, class_idx):
        """Map a class index to its use case name"""
        # Find the use case that maps to this index
        for use_case, idx in self.label_mapping.items():
            if idx == class_idx:
                return use_case
        return f"Unknown_Class_{class_idx}"
    
    def save_predictions(self, simplified_results, output_path):
        """
        Save simplified predictions to CSV
        
        Args:
            simplified_results: DataFrame with SMILES and predictions
            output_path: Path where to save the CSV file
        """
        simplified_results.to_csv(output_path, index=False)
        print(f"Saved simplified predictions to {output_path}")

# Example usage
if __name__ == "__main__":
    # Initialize predictor with the path to your trained model directory and label mapping
    predictor = MoleculePredictor(
        model_dir="../../../Results/NNTrainingResults/20250121_104305",
        label_mapping_path="../../../QSUR/FunctionalUseEncoding.json"
    )
    
    # Load test data
    test_data = pd.read_csv("../../../Data/Curated/UseCaseDataModeling.csv", low_memory=False)
    
    # Make predictions
    detailed_predictions, simplified_results = predictor.predict(test_data, smiles_column='SMILES')
    
    # Save simplified results
    output_file = "PredictionsOnTestSet.csv"
    predictor.save_predictions(simplified_results, output_file)