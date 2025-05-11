"""
This is for trainign individual Isolation forests to predict moelcular use-cases 

Input: 2048 Bit-cvectors
Output: Strength of assocaition to use-case

"""
# import Statments
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score, roc_curve, auc
from joblib import dump, load
import os
import json
from typing import Tuple, Dict, Union, List
from pathlib import Path

#class with all the functions I need for training

class IsolationForestTrainer:
    def __init__(self, 
                 n_estimators: int = 100, 
                 contamination: float = 'auto',
                 random_state: int = 42,
                 feature_prefix: str = 'Bit'):
        """
        Initialize the Isolation Forest classifier.
        
        Args:
            n_estimators: Number of estimators in the forest
            contamination: Expected proportion of outliers in the dataset
            random_state: Random state for reproducibility
            feature_prefix: Prefix for feature columns
        """
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state
        )
        self.feature_prefix = feature_prefix
        self.threshold = None
        self.feature_columns = None

    def prepare_data(self, data: pd.DataFrame, class_column: str, use_case: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training and testing.
        
        Args:
            data: Input DataFrame
            class_column: Name of the class column
            use_case: Specific use case to filter for
            
        Returns:
            X: Features DataFrame
            y: Labels Series
        """
        df = data[data[class_column] == use_case].copy()
        y = df[class_column]
        X = df.filter(regex=f'^{self.feature_prefix}')
        self.feature_columns = X.columns.tolist()
        return X, y
    
    def train(self, X: pd.DataFrame, custom_threshold: float = None) -> Dict[str, float]:
        """
        Train the Isolation Forest model.
        
        Args:
            X: Training features
            custom_threshold: Optional custom threshold for anomaly detection
            
        Returns:
            Dict containing training metrics
        """
        self.model.fit(X)
        scores = self.model.decision_function(X)
        self.threshold = custom_threshold if custom_threshold is not None else np.median(scores)
        
        # Calculate basic training metrics
        predictions = self.predict(X)
        metrics = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'threshold': self.threshold
        }
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features DataFrame
            
        Returns:
            numpy array of predictions (-1 for anomalies, 1 for normal)
        """
        if not self.feature_columns:
            raise ValueError("Model hasn't been trained yet - no feature columns saved")
        
        if not all(col in X.columns for col in self.feature_columns):
            raise ValueError("Input data doesn't match training features")
            
        return self.model.predict(X)
    
    def save_model(self, path: Union[str, Path]) -> None:
        """
        Save the trained model and associated metadata.
        
        Args:
            path: Path to save the model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'threshold': self.threshold,
            'feature_columns': self.feature_columns,
            'feature_prefix': self.feature_prefix
        }
        dump(model_data, path)
    
    def load_model(self, path: Union[str, Path]) -> None:
        """
        Load a trained model and associated metadata.
        
        Args:
            path: Path to the saved model
        """
        model_data = load(path)
        self.model = model_data['model']
        self.threshold = model_data['threshold']
        self.feature_columns = model_data['feature_columns']
        self.feature_prefix = model_data['feature_prefix']