import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import os
from datetime import datetime
import json
from pathlib import Path

class Config:
    def __init__(self, **kwargs):
        # Data parameters
        self.data_path = kwargs.get('data_path', 'data/UseCaseDataModeling.csv')
        self.min_samples = kwargs.get('min_samples', 400)
        self.batch_size = kwargs.get('batch_size', 32)
        self.feature_prefix = kwargs.get('feature_prefix', 'Bit_')
        self.target_column = kwargs.get('target_column', 'Harmonized Functional Use Encoded')
        self.name_column = kwargs.get('name_column', 'Harmonized Functional Use')
        
        # Model parameters
        self.hidden_dims = kwargs.get('hidden_dims', [1024, 512, 256])
        self.dropout_rate = kwargs.get('dropout_rate', 0.2)
        
        # Training parameters
        self.learning_rate = kwargs.get('learning_rate', 0.0001)
        self.num_epochs = kwargs.get('num_epochs', 50)
        self.patience = kwargs.get('patience', 15)
        
        # Output parameters
        self.output_dir = kwargs.get('output_dir', 'outputs')
        self.run_name = kwargs.get('run_name', datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        # Create output directories
        self.model_dir = os.path.join(self.output_dir, self.run_name, 'models')
        self.plot_dir = os.path.join(self.output_dir, self.run_name, 'plots')
        self.create_directories()
        
        # Save configuration
        self.save_config()
    
    def create_directories(self):
        """Create necessary directories for outputs"""
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
    
    def save_config(self):
        """Save configuration to JSON file"""
        config_path = os.path.join(self.output_dir, self.run_name, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

class MoleculeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DataManager:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.data_info = {}
    
    def prepare_data(self):
        df = pd.read_csv(self.config.data_path)
        
        # Separate features and target
        X = df.filter(regex=f'^{self.config.feature_prefix}')
        y = df[self.config.target_column]
        original_names = df[self.config.name_column]
        
        # Filter classes and create mappings
        filtered_data = self._filter_classes(X, y, original_names)
        if filtered_data is None:
            raise ValueError("No classes meet the minimum sample requirement")
        
        X, y, original_names, label_mapping = filtered_data
        
        # Split and scale data
        train_loader, val_loader, test_loader = self._split_and_scale_data(X, y)
        
        # Store data info
        self.data_info.update({
            'input_dim': X.shape[1],
            'num_classes': len(y.unique()),
            'label_mapping': label_mapping,
            'reverse_mapping': {v: k for k, v in label_mapping.items()},
            'original_functions': original_names.unique()
        })
        
        return train_loader, val_loader, test_loader
    
    def _filter_classes(self, X, y, original_names):
        class_counts = y.value_counts()
        valid_classes = class_counts[class_counts >= self.config.min_samples].index
        
        if len(valid_classes) == 0:
            return None
        
        mask = y.isin(valid_classes)
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
        original_names = original_names[mask].reset_index(drop=True)
        
        # Create label mappings
        class_mapping = {old: new for new, old in enumerate(sorted(y.unique()))}
        y = y.map(class_mapping)
        
        label_mapping = {}
        for new_label in y.unique():
            old_label = [k for k, v in class_mapping.items() if v == new_label][0]
            label_mapping[new_label] = original_names[y == new_label].iloc[0]
        
        return X, y, original_names, label_mapping
    
    def _split_and_scale_data(self, X, y):
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Calculate class weights for balanced sampling
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        class_weights = total_samples / (len(class_counts) * class_counts)
        sample_weights = class_weights[y_train]
        
        # Create sampler for balanced batches
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(y_train),
            replacement=True
        )
        
        # Create datasets
        train_dataset = MoleculeDataset(X_train_scaled, y_train.values)
        val_dataset = MoleculeDataset(X_val_scaled, y_val.values)
        test_dataset = MoleculeDataset(X_test_scaled, y_test.values)
        
        # Create dataloaders - note the sampler is only used for training
        return (
            DataLoader(train_dataset, batch_size=self.config.batch_size, sampler=sampler),
            DataLoader(val_dataset, batch_size=self.config.batch_size),
            DataLoader(test_dataset, batch_size=self.config.batch_size)
        )

class MoleculeNet(nn.Module):
    def __init__(self, config, input_dim, num_classes):
        super(MoleculeNet, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class Trainer:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        # Add learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
    def train(self, train_loader, val_loader):
        best_val_accuracy = 0
        patience_counter = 0
        train_losses, val_losses = [], []
        
        for epoch in range(self.config.num_epochs):
            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc = self._validate(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f'Epoch {epoch+1}/{self.config.num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter = 0
                self._save_checkpoint(epoch, val_acc)
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print("Early stopping triggered")
                    break
        
        # Plot training curves
        self._plot_training_curves(train_losses, val_losses)
    
    def _train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return total_loss/len(train_loader), correct/total
    
    def _validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return total_loss/len(val_loader), correct/total
    
    def _save_checkpoint(self, epoch, val_acc):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_accuracy': val_acc
        }
        path = os.path.join(self.config.model_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
    
    def _plot_training_curves(self, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.config.plot_dir, 'training_curves.png'))
        plt.close()

class MetricsPlotter:
    def __init__(self, model, data_info, config, device):
        self.model = model
        self.data_info = data_info
        self.config = config
        self.device = device
    
    def plot_metrics(self, test_loader):
        y_score, y_true = self._get_predictions(test_loader)
        
        self._plot_precision_recall_curves(y_true, y_score)
        self._plot_precision_recall_heatmap(y_true, y_score)
        self._plot_confusion_matrix(y_true, y_score)
    
    def _get_predictions(self, test_loader):
        self.model.eval()
        y_score = []
        y_true = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                probas = torch.nn.functional.softmax(outputs, dim=1)
                
                y_score.append(probas.cpu().numpy())
                y_true.append(labels.cpu().numpy())
        
        return np.vstack(y_score), np.hstack(y_true)
    
    def _plot_precision_recall_curves(self, y_true, y_score):
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        # Create a figure to show class distribution
        plt.figure(figsize=(8, 4))
        class_counts = np.bincount(y_true)
        plt.bar(range(len(class_counts)), class_counts)
        plt.title('Class Distribution in Test Set')
        plt.xlabel('Class Index')
        plt.ylabel('Number of Samples')
        plt.savefig(os.path.join(self.config.plot_dir, 'class_distribution.png'))
        plt.close()
        
        # Plot individual PR curves
        for i in range(self.data_info['num_classes']):
            plt.figure(figsize=(8, 6))
            true_label = (y_true == i)
            class_score = y_score[:, i]
            
            precision, recall, _ = precision_recall_curve(true_label, class_score)
            avg_precision = average_precision_score(true_label, class_score)
            
            # Calculate no-skill line (proportion of positive class)
            no_skill = np.sum(true_label) / len(true_label)
            
            # Plot the curves
            plt.plot(recall, precision, lw=2, 
                    label=f'AP = {avg_precision:.2f}\nSupport = {np.sum(true_label)}')
            plt.plot([0, 1], [no_skill, no_skill], '--', 
                    label=f'No Skill (support ratio = {no_skill:.3f})')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {self.data_info["label_mapping"][i]}')
            plt.legend(loc="best")
            plt.grid(True)
            plt.savefig(os.path.join(self.config.plot_dir, f'pr_curve_class_{i}.png'))
            plt.close()
    
    def _plot_precision_recall_heatmap(self, y_true, y_score):
        _, predicted = torch.max(torch.tensor(y_score), 1)
        y_pred = predicted.numpy()
        
        precision, recall, _, support = precision_recall_fscore_support(y_true, y_pred)
        metrics_matrix = np.column_stack((precision, recall))
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(metrics_matrix,
                   annot=True,
                   fmt='.3f',
                   cmap='plasma',
                   xticklabels=['Precision', 'Recall', 'Support'],
                   yticklabels=[self.data_info["label_mapping"][i] for i in range(len(precision))],
                   cbar_kws={'label': 'Score'})
        plt.title('Precision, Recall, and Support by Class')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plot_dir, 'precision_recall_heatmap.png'))
        plt.close()
    
    def _plot_confusion_matrix(self, y_true, y_score):
        _, predicted = torch.max(torch.tensor(y_score), 1)
        y_pred = predicted.numpy()
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm,
                   annot=True,
                   fmt='d',
                   cmap='plasma',
                   xticklabels=[self.data_info["label_mapping"][i] for i in range(cm.shape[0])],
                   yticklabels=[self.data_info["label_mapping"][i] for i in range(cm.shape[0])])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plot_dir, 'confusion_matrix.png'))
        plt.close()

def main():
    # Set up configuration
    config = Config(
        data_path='../../../Data/Curated/UseCaseDataModeling.csv',
        output_dir='../../../Results/NNTrainingResults',
        hidden_dims=[512, 256, 128],
        batch_size=32,
        num_epochs=50,
        patience=15
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    trainer.train(train_loader, val_loader)
    
    # Load best model and generate metrics
    best_checkpoint = max(Path(config.model_dir).glob('checkpoint_*.pt'), 
                         key=lambda p: float(p.stem.split('_')[2]))
    checkpoint = torch.load(best_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Plot metrics
    metrics_plotter = MetricsPlotter(model, data_manager.data_info, config, device)
    metrics_plotter.plot_metrics(test_loader)

if __name__ == '__main__':
    main()