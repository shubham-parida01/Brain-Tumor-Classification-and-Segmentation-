"""
Comprehensive metrics for brain tumor detection evaluation.
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from scipy.spatial import distance
import cv2

class ClassificationMetrics:
    """Metrics for brain tumor classification."""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
        
    def update(self, preds, targets, probs):
        """Update metrics with new predictions."""
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.probabilities.extend(probs.cpu().numpy())
        
    def compute(self):
        """Compute all classification metrics."""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities)
        
        # Basic metrics
        conf_matrix = confusion_matrix(targets, predictions)
        class_report = classification_report(targets, predictions, output_dict=True)
        
        # ROC and AUC
        roc_curves = {}
        auc_scores = {}
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(targets == i, probabilities[:, i])
            roc_curves[i] = (fpr, tpr)
            auc_scores[i] = auc(fpr, tpr)
        
        # Calculate per-class metrics
        metrics = {
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'roc_curves': roc_curves,
            'auc_scores': auc_scores,
            'mean_auc': np.mean(list(auc_scores.values())),
            'accuracy': class_report['accuracy'],
            'macro_precision': class_report['macro avg']['precision'],
            'macro_recall': class_report['macro avg']['recall'],
            'macro_f1': class_report['macro avg']['f1-score'],
            'weighted_precision': class_report['weighted avg']['precision'],
            'weighted_recall': class_report['weighted avg']['recall'],
            'weighted_f1': class_report['weighted avg']['f1-score']
        }
        
        return metrics

class SegmentationMetrics:
    """Metrics for brain tumor segmentation."""
    
    def __init__(self, num_classes=1):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        
    def update(self, preds, targets):
        """Update metrics with new predictions."""
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        
    def compute(self):
        """Compute all segmentation metrics."""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {}
        
        # Dice coefficient
        dice_scores = []
        for pred, target in zip(predictions, targets):
            intersection = np.sum(pred * target)
            dice = (2. * intersection) / (np.sum(pred) + np.sum(target) + 1e-6)
            dice_scores.append(dice)
        metrics['dice'] = np.mean(dice_scores)
        
        # IoU (Intersection over Union)
        iou_scores = []
        for pred, target in zip(predictions, targets):
            intersection = np.sum(pred * target)
            union = np.sum(pred) + np.sum(target) - intersection
            iou = intersection / (union + 1e-6)
            iou_scores.append(iou)
        metrics['iou'] = np.mean(iou_scores)
        
        # Precision and Recall
        precision_scores = []
        recall_scores = []
        for pred, target in zip(predictions, targets):
            tp = np.sum(pred * target)
            fp = np.sum(pred * (1 - target))
            fn = np.sum((1 - pred) * target)
            
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            
            precision_scores.append(precision)
            recall_scores.append(recall)
        metrics['precision'] = np.mean(precision_scores)
        metrics['recall'] = np.mean(recall_scores)
        
        # F1 Score
        metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'] + 1e-6)
        
        # Hausdorff Distance
        hausdorff_scores = []
        for pred, target in zip(predictions, targets):
            pred_points = np.argwhere(pred > 0.5)
            target_points = np.argwhere(target > 0.5)
            
            if len(pred_points) > 0 and len(target_points) > 0:
                dist = distance.directed_hausdorff(pred_points, target_points)[0]
                hausdorff_scores.append(dist)
        metrics['hausdorff'] = np.mean(hausdorff_scores) if hausdorff_scores else float('inf')
        
        # Boundary IoU
        boundary_iou_scores = []
        for pred, target in zip(predictions, targets):
            pred_boundary = cv2.Canny(pred.astype(np.uint8), 100, 200)
            target_boundary = cv2.Canny(target.astype(np.uint8), 100, 200)
            
            intersection = np.sum(pred_boundary * target_boundary)
            union = np.sum(pred_boundary) + np.sum(target_boundary) - intersection
            boundary_iou = intersection / (union + 1e-6)
            boundary_iou_scores.append(boundary_iou)
        metrics['boundary_iou'] = np.mean(boundary_iou_scores)
        
        return metrics

def compute_combined_metrics(class_metrics, seg_metrics):
    """Compute combined metrics for both classification and segmentation."""
    combined = {}
    
    # Classification metrics
    for k, v in class_metrics.items():
        combined[f'class_{k}'] = v
    
    # Segmentation metrics
    for k, v in seg_metrics.items():
        combined[f'seg_{k}'] = v
    
    # Combined score (weighted average of key metrics)
    combined['combined_score'] = (
        0.4 * class_metrics['accuracy'] +
        0.3 * class_metrics['mean_auc'] +
        0.2 * seg_metrics['dice'] +
        0.1 * seg_metrics['boundary_iou']
    )
    
    return combined

def compute_metrics_from_outputs(outputs, targets):
    """
    Compute metrics from model outputs and targets.
    
    Args:
        outputs (torch.Tensor): Model outputs
        targets (torch.Tensor): Ground truth labels
        
    Returns:
        tuple: (metrics_dict, predictions, probabilities)
    """
    # Convert to numpy
    outputs = outputs.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Get predictions and probabilities
    predictions = np.argmax(outputs, axis=1)
    probabilities = torch.nn.functional.softmax(torch.from_numpy(outputs), dim=1).numpy()
    
    # Compute metrics
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = np.mean(predictions == targets)
    
    # Precision, Recall, F1
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='weighted')
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    # ROC AUC
    from sklearn.metrics import roc_auc_score
    try:
        metrics['roc_auc'] = roc_auc_score(targets, probabilities, multi_class='ovr')
    except ValueError:
        metrics['roc_auc'] = 0.0
    
    return metrics, predictions, probabilities