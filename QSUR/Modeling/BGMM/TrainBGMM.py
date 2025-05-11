import pandas as pd
import numpy as np
import umap
import pickle
import os
from sklearn.mixture import BayesianGaussianMixture
from pathlib import Path

class UMAPBGMMPipeline:
    def __init__(self, umap_dims=50, n_components=15, random_state=42):
        """
        Initialize the UMAP-BGMM pipeline.
        
        Args:
            umap_dims (int): Number of dimensions for UMAP reduction
            n_components (int): Number of components for BGMM
            random_state (int): Random state for reproducibility
        """
        self.umap_dims = umap_dims
        self.n_components = n_components
        self.random_state = random_state
        self.models = {}
        
    def fit(self, data, bit_columns=None, use_column='Harmonized Functional Use Encoded', save_dir=None):
        """
        Fit the UMAP-BGMM pipeline on the data.
        
        Args:
            data (pd.DataFrame): Input DataFrame
            bit_columns (list): List of bit columns to use. If None, uses columns starting with 'Bit'
            use_column (str): Column name containing use cases
            save_dir (str): Directory to save models. If None, models are only kept in memory
        """
        if bit_columns is None:
            bit_columns = [col for col in data.columns if col.startswith('Bit')]
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        results = []
        unique_uses = data[use_column].unique()
        
        for use in unique_uses:
            try:
                print(f"Processing use case {use}")
                use_data = data[data[use_column] == use]
                X = use_data[bit_columns]
                
                if len(X) == 0:
                    print(f"Skipping use {use}: empty dataset")
                    continue
                
                # Fit UMAP
                reducer = umap.UMAP(n_components=self.umap_dims, random_state=self.random_state)
                reduced_data = reducer.fit_transform(X)
                
                # Fit BGMM
                bgmm = BayesianGaussianMixture(
                    n_components=self.n_components,
                    random_state=self.random_state,
                    max_iter=1000,
                    n_init=5
                )
                bgmm.fit(reduced_data)
                
                # Store models
                self.models[use] = {
                    'umap': reducer,
                    'bgmm': bgmm
                }
                
                # Save models if directory is provided
                if save_dir:
                    with open(os.path.join(save_dir, f'models_{use}.pkl'), 'wb') as f:
                        pickle.dump(self.models[use], f)
                
                # Calculate metrics
                n_samples = len(reduced_data)
                n_features = reduced_data.shape[1]
                loglik = bgmm.score(reduced_data) * n_samples
                n_parameters = (self.n_components * n_features * (n_features + 1) / 2) + \
                             (self.n_components * n_features) + (self.n_components - 1)
                
                results.append({
                    'use_case': use,
                    'n_samples': n_samples,
                    'log_likelihood': loglik,
                    'aic': -2 * loglik + 2 * n_parameters,
                    'bic': -2 * loglik + n_parameters * np.log(n_samples),
                    'converged': bgmm.converged_
                })
                
            except Exception as e:
                print(f"Error processing use {use}: {str(e)}")
        
        return pd.DataFrame(results)
    
    @classmethod
    def load_models(cls, model_dir):
        """
        Load previously saved models.
        
        Args:
            model_dir (str): Directory containing saved models
            
        Returns:
            UMAPBGMMPipeline: Instance with loaded models
        """
        instance = cls()
        model_files = Path(model_dir).glob('models_*.pkl')
        
        for model_file in model_files:
            use = int(model_file.stem.split('_')[1])
            with open(model_file, 'rb') as f:
                instance.models[use] = pickle.load(f)
        
        return instance
    
    def transform(self, data, use_case, bit_columns=None):
        """
        Transform new data using fitted models for a specific use case.
        
        Args:
            data (pd.DataFrame): Input data
            use_case: Use case identifier
            bit_columns (list): List of bit columns to use. If None, uses columns starting with 'Bit'
            
        Returns:
            tuple: (UMAP transformed data, BGMM predictions)
        """
        if bit_columns is None:
            bit_columns = [col for col in data.columns if col.startswith('Bit')]
        
        if use_case not in self.models:
            raise ValueError(f"No models found for use case {use_case}")
        
        X = data[bit_columns]
        reduced_data = self.models[use_case]['umap'].transform(X)
        predictions = self.models[use_case]['bgmm'].predict(reduced_data)
        
        return reduced_data, predictions


# Example usage:
if __name__ == "__main__":
    # Load your data
    data = pd.read_csv('../../../Data/Curated/UseCaseDataModeling.csv')
    
    # Create and fit the pipeline
    pipeline = UMAPBGMMPipeline(umap_dims=50, n_components=15)
    results = pipeline.fit(data, save_dir='/home/matt/Proj/QSURv3/QSUR/Models/BGMMs')
    results.to_csv('../../../Results/BGMM/fitting_results.csv', index=False)
"""
    # Later, load and use saved models
    loaded_pipeline = UMAPBGMMPipeline.load_models('/home/matt/Proj/QSURv3/QSUR/Models/BGMMs')
    
    # Transform new data for a specific use case
    new_data = pd.read_csv('path_to_new_data.csv')
    reduced_data, predictions = loaded_pipeline.transform(new_data, use_case=0)
"""