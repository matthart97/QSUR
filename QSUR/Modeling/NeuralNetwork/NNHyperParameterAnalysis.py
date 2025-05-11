import torch
import json
from pathlib import Path
import glob
import pandas as pd
import re

# Import your model class
from TrainNN import MoleculeNet, DataManager, Config

def find_latest_checkpoint(run_path):
    """Find the latest checkpoint in a run directory"""
    checkpoints = list(Path(run_path).glob('**/checkpoint_epoch_*.pt'))
    if not checkpoints:
        return None
    
    # Extract epoch numbers and find latest
    epoch_nums = [int(re.search(r'epoch_(\d+)\.pt', str(cp)).group(1)) for cp in checkpoints]
    latest_idx = max(range(len(epoch_nums)), key=epoch_nums.__getitem__)
    return checkpoints[latest_idx]

def load_run_info(run_path):
    """Load config and model from a run directory"""
    # Find latest timestamp directory
    timestamp_dirs = list(Path(run_path).glob('2*'))
    if not timestamp_dirs:
        return None
    latest_dir = max(timestamp_dirs)
    
    # Load config
    config_path = latest_dir / 'config.json'
    if not config_path.exists():
        return None
        
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Find latest checkpoint
    checkpoint_path = find_latest_checkpoint(latest_dir)
    if checkpoint_path is None:
        return None
        
    return {
        'config': config_dict,
        'checkpoint_path': checkpoint_path,
        'run_dir': run_path
    }

def evaluate_model(model, data_loader, device):
    """Evaluate model on given data loader"""
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    avg_loss = val_loss / len(data_loader)
    return avg_loss, accuracy

def analyze_runs(tuning_dir, data_path):
    """Analyze all runs in the tuning directory"""
    tuning_dir = Path(tuning_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Find all run directories
    run_dirs = sorted(tuning_dir.glob('run_*'))
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {tuning_dir}")
    
    print(f"Found {len(run_dirs)} run directories")
    
    # Prepare data once
    base_config = Config(data_path=data_path)
    data_manager = DataManager(base_config)
    _, val_loader, _ = data_manager.prepare_data()
    
    results = []
    for run_dir in run_dirs:
        run_number = int(run_dir.name.split('_')[1])
        print(f"\nProcessing run {run_number}")
        
        run_info = load_run_info(run_dir)
        if run_info is None:
            print(f"Skipping run {run_number} - missing files")
            continue
            
        try:
            # Create config and model
            config_dict = run_info['config']
            config = Config(**config_dict)
            
            model = MoleculeNet(
                config=config,
                input_dim=data_manager.data_info['input_dim'],
                num_classes=data_manager.data_info['num_classes']
            ).to(device)
            
            # Load checkpoint
            checkpoint = torch.load(run_info['checkpoint_path'], map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Evaluate model
            val_loss, val_accuracy = evaluate_model(model, val_loader, device)
            
            # Store results
            result = {
                'run': run_number,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'learning_rate': config.learning_rate,
                'hidden_dims': str(config.hidden_dims),
                'dropout_rate': config.dropout_rate,
                'batch_size': config.batch_size,
                'checkpoint': str(run_info['checkpoint_path'])
            }
            results.append(result)
            
            print(f"Run {run_number} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
        except Exception as e:
            print(f"Error processing run {run_number}: {str(e)}")
            continue
    
    # Convert to DataFrame and find best run
    results_df = pd.DataFrame(results)
    results_df.to_csv(tuning_dir / 'reconstructed_results.csv', index=False)
    
    if len(results_df) > 0:
        best_run = results_df.loc[results_df['val_accuracy'].idxmax()]
        
        print("\nBest parameters found:")
        for param, value in best_run.items():
            if param not in ['checkpoint']:
                print(f"{param}: {value}")
        
        # Save best parameters
        with open(tuning_dir / 'best_parameters.json', 'w') as f:
            json.dump(best_run.to_dict(), f, indent=4)
        
        print(f"\nResults saved to {tuning_dir / 'reconstructed_results.csv'}")
        print(f"Best parameters saved to {tuning_dir / 'best_parameters.json'}")
        
        return best_run.to_dict()
    else:
        print("No valid results found")
        return None

if __name__ == "__main__":
    tuning_dir = "../../../Results/NNTrainingResults/NNHyperparameterTuning/hyperparameter_tuning_20250121_105035"
    data_path = "../../../Data/Curated/UseCaseDataModeling.csv"
    
    best_params = analyze_runs(tuning_dir, data_path)