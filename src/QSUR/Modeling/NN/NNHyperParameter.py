import itertools
import json
from pathlib import Path
import torch
from datetime import datetime
import pandas as pd

# Import from your training script
from TrainNN import Config, DataManager, MoleculeNet, Trainer, MetricsPlotter

def grid_search_hyperparameters(base_data_path, base_output_dir):
    # Define hyperparameter grid
    param_grid = {
        'learning_rate': [0.01, 0.001, 0.0001],
        'hidden_dims': [
            [512, 256, 128],
            [1024, 512, 256],
            [256, 128, 64]
        ],
        'dropout_rate': [0.2, 0.3, 0.4],
        'batch_size': [32, 64],
    }
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tuning_dir = Path(base_output_dir) / f"hyperparameter_tuning_{timestamp}"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    
    # Store all results
    results = []
    
    # Generate all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Save hyperparameter grid for reference
    with open(tuning_dir / 'param_grid.json', 'w') as f:
        json.dump(param_grid, f, indent=4)
    
    print(f"Starting grid search with {len(combinations)} combinations")
    
    # Try each combination
    for i, params in enumerate(combinations, 1):
        print(f"\nTrying combination {i}/{len(combinations)}:")
        print(params)
        
        # Create run directory
        run_dir = tuning_dir / f"run_{i}"
        
        # Create config with current parameters
        config = Config(
            data_path=base_data_path,
            output_dir=str(run_dir),
            hidden_dims=params['hidden_dims'],
            dropout_rate=params['dropout_rate'],
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size'],
            num_epochs=50,
            patience=15
        )
        
        try:
            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Prepare data
            data_manager = DataManager(config)
            train_loader, val_loader, test_loader = data_manager.prepare_data()
            
            # Initialize model
            model = MoleculeNet(
                config=config,
                input_dim=data_manager.data_info['input_dim'],
                num_classes=data_manager.data_info['num_classes']
            ).to(device)
            
            # Train model
            trainer = Trainer(model, config, device)
            best_val_loss, best_val_acc = trainer.train(train_loader, val_loader)
            
            # Generate metrics
            metrics_plotter = MetricsPlotter(model, data_manager.data_info, config, device)
            metrics_plotter.plot_metrics(test_loader)
            
            # Store results
            result = {
                'run': i,
                'val_loss': best_val_loss,
                'val_accuracy': best_val_acc,
                **params
            }
            results.append(result)
            
            # Save current results to CSV
            results_df = pd.DataFrame(results)
            results_df.to_csv(tuning_dir / 'results.csv', index=False)
            
            print(f"Run {i} completed - Val Loss: {best_val_loss:.4f}, Val Accuracy: {best_val_acc:.4f}")
            
        except Exception as e:
            print(f"Error in run {i}: {str(e)}")
            continue
    
    # Find best parameters
    results_df = pd.DataFrame(results)
    best_run = results_df.loc[results_df['val_accuracy'].idxmax()]
    
    print("\nGrid Search completed!")
    print("\nBest parameters:")
    for param, value in best_run.items():
        if param not in ['run', 'val_loss', 'val_accuracy']:
            print(f"{param}: {value}")
    print(f"Best validation accuracy: {best_run['val_accuracy']:.4f}")
    print(f"Best validation loss: {best_run['val_loss']:.4f}")
    
    # Save final summary
    with open(tuning_dir / 'best_parameters.json', 'w') as f:
        json.dump(best_run.to_dict(), f, indent=4)
    
    return best_run.to_dict()

if __name__ == "__main__":
    # Paths to your data and output directory
    base_data_path = "/home/matt/Proj/QSURv3/Data/Curated/UseCaseDataModeling.csv"
    base_output_dir = "/home/matt/Proj/QSURv3/Results/NNTrainingResults/NNHyperparameterTuning"
    
    # Run hyperparameter tuning
    best_params = grid_search_hyperparameters(base_data_path, base_output_dir)