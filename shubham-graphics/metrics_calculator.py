import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import torch
from scipy.stats import ttest_ind

class MetricsCalculator:
    def __init__(self):
        self.metrics = {
            'accuracy': [],
            'sensitivity': [],
            'specificity': [],
            'auc': []
        }
        
    def calculate_metrics(self, y_true, y_pred, y_prob):
        # Convert to numpy if tensors
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        if torch.is_tensor(y_prob):
            y_prob = y_prob.cpu().numpy()
            
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate metrics
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        auc = roc_auc_score(y_true, y_prob)
        
        # Store metrics
        self.metrics['accuracy'].append(accuracy)
        self.metrics['sensitivity'].append(sensitivity)
        self.metrics['specificity'].append(specificity)
        self.metrics['auc'].append(auc)
        
        return accuracy, sensitivity, specificity, auc
        
    def get_summary_statistics(self):
        """Calculate mean and standard deviation for all metrics"""
        summary = {}
        
        for metric, values in self.metrics.items():
            mean = np.mean(values)
            std = np.std(values)
            summary[metric] = f"{mean:.3f} ± {std:.3f}"
            
        return summary
        
    def print_results(self):
        """Print results in a format similar to the paper"""
        summary = self.get_summary_statistics()
        
        print("\nClassification Results:")
        print("Method\tAccuracy\tSensitivity\tSpecificity\tAUC")
        print("CNN")
        print(f"Average\t{summary['accuracy']}\t"
              f"{summary['sensitivity']}\t"
              f"{summary['specificity']}\t"
              f"{summary['auc']}") 