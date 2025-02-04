import torch
import pandas as pd
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import numpy as np
from TrainNN import MoleculeNet, Config
from torch.utils.data import DataLoader, TensorDataset
import gc
import os
from datetime import datetime

class MoleculePredictor:
    def __init__(self, model_dir, label_mapping_path, batch_size=1024):
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
        # Load label mapping
        with open(label_mapping_path, 'r') as f:
            self.label_mapping = json.load(f)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize model
        self.model = self._load_model()
        self.scaler = None  # Will be fit during first chunk processing
        
        print(f"Model loaded successfully from {model_dir}")
        print(f"Label mapping loaded from {label_mapping_path}")
        print(f"Using device: {self.device}")
        print(f"Batch size: {self.batch_size}")

    def _load_config(self):
        config_path = self.model_dir / 'config.json'
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return Config(**config_dict)

    def _load_model(self):
        checkpoint_path = max(self.model_dir.glob('models/checkpoint_*.pt'),
                            key=lambda p: float(p.stem.split('_')[2]))
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        output_size = checkpoint['model_state_dict']['model.9.weight'].shape[0]
        
        model = MoleculeNet(
            config=self.config,
            input_dim=self._get_input_dim(),
            num_classes=output_size
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def _get_input_dim(self):
        df = pd.read_csv(self.config.data_path, nrows=1, low_memory=False)
        return len(df.filter(regex=f'^{self.config.feature_prefix}').columns)

    def _setup_output_directory(self, base_output_path):
        """Create output directory with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(base_output_path) / f"predictions_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def process_chunk(self, chunk, smiles_column='SMILES'):
        """Process a single chunk of data"""
        features = chunk.filter(regex=f'^{self.config.feature_prefix}')
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(features)
        else:
            scaled_features = self.scaler.transform(features)

        dataset = TensorDataset(torch.FloatTensor(scaled_features))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for batch in dataloader:
                batch_features = batch[0].to(self.device)
                outputs = self.model(batch_features)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        predicted_uses = [self.get_use_case(idx) for idx in all_predictions]
        
        chunk_results = pd.DataFrame({
            'SMILES': chunk[smiles_column],
            'Predicted_Class_Index': all_predictions,
            'Predicted_Use': predicted_uses,
            'Confidence': [probs[pred] for probs, pred in zip(all_probabilities, all_predictions)]
        })

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return chunk_results

    def predict(self, input_path, output_path, smiles_column='SMILES', chunk_size=10000, save_interval=10000):
        """
        Process large datasets in chunks and save results incrementally
        
        Args:
            input_path: Path to input CSV file
            output_path: Base path for output directory
            smiles_column: Name of the SMILES column
            chunk_size: Number of rows to process at once
            save_interval: Number of predictions after which to save a new file
        """
        # Create output directory
        output_dir = self._setup_output_directory(output_path)
        print(f"Saving results to directory: {output_dir}")

        # Initialize counters
        total_predictions = 0
        current_batch_results = []
        file_counter = 1

        # Process data in chunks
        chunk_iterator = pd.read_csv(input_path, chunksize=chunk_size, low_memory=False)
        
        for i, chunk in enumerate(chunk_iterator, 1):
            print(f"Processing chunk {i}...")
            
            # Process chunk
            chunk_results = self.process_chunk(chunk, smiles_column)
            current_batch_results.append(chunk_results)
            total_predictions += len(chunk_results)
            
            # Save results if we've reached the interval or it's the last chunk
            if total_predictions >= save_interval * file_counter:
                # Combine all accumulated results
                combined_results = pd.concat(current_batch_results, ignore_index=True)
                
                # Save to file
                output_file = output_dir / f"predictions_batch_{file_counter}.csv"
                combined_results.to_csv(output_file, index=False)
                print(f"Saved {len(combined_results)} predictions to {output_file}")
                
                # Clear the batch results list and increment counter
                current_batch_results = []
                file_counter += 1
            
            # Force garbage collection
            gc.collect()
            
            if i % 10 == 0:
                print(f"Processed {total_predictions} rows...")

        # Save any remaining results
        if current_batch_results:
            combined_results = pd.concat(current_batch_results, ignore_index=True)
            output_file = output_dir / f"predictions_batch_{file_counter}.csv"
            combined_results.to_csv(output_file, index=False)
            print(f"Saved final {len(combined_results)} predictions to {output_file}")

        # Save a manifest file with processing details
        manifest = {
            'total_predictions': total_predictions,
            'number_of_files': file_counter,
            'chunk_size': chunk_size,
            'save_interval': save_interval,
            'timestamp': datetime.now().isoformat(),
            'input_file': str(input_path)
        }
        
        with open(output_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)

        # Combine all files into one final output
        print("Creating combined output file...")
        all_files = sorted(output_dir.glob('predictions_batch_*.csv'))
        
        # Read and combine all batch files
        combined_df = pd.concat(
            (pd.read_csv(f) for f in all_files),
            ignore_index=True
        )
        
        # Save combined results
        combined_output = output_dir / 'predictions_combined.csv'
        combined_df.to_csv(combined_output, index=False)
        
        # Update manifest
        manifest['combined_file'] = str(combined_output)
        with open(output_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
            
        print(f"Processing complete. Results saved in {output_dir}")
        print(f"Combined results saved to {combined_output}")
        return output_dir

    def get_use_case(self, class_idx):
        """Map a class index to its use case name"""
        # Find the use case that maps to this index
        for use_case, idx in self.label_mapping.items():
            if idx == class_idx:
                return use_case
        return f"Unknown_Class_{class_idx}"

# Example usage
if __name__ == "__main__":
    predictor = MoleculePredictor(
        model_dir="/home/matt/Proj/QSURv3/Results/NNTrainingResults/20250121_104305",
        label_mapping_path="/home/matt/Proj/QSURv3/QSUR/FunctionalUseEncoding.json",
        batch_size=1024
    )
    
    predictor.predict(
        input_path="/home/matt/Proj/QSURv3/PatentMolProcessing/PatentSceeningSet.csv",
        output_path="prediction_results",  # This will create a timestamped directory
        chunk_size=10000,
        save_interval=10000  # Save a new file every 10000 predictions
    )