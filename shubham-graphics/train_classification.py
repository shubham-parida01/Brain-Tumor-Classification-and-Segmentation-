import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from metrics_calculator import MetricsCalculator
import numpy as np

class ClassificationModel(nn.Module):
    def __init__(self, num_classes=4):
        super(ClassificationModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    metrics_calculator = MetricsCalculator()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Get predictions and probabilities
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        # Calculate training metrics
        train_metrics = metrics_calculator.calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        val_probs = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                probs = torch.softmax(outputs, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
        
        # Calculate validation metrics
        val_metrics = metrics_calculator.calculate_metrics(
            np.array(val_labels),
            np.array(val_preds),
            np.array(val_probs)
        )
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {running_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
        print('Training Metrics:')
        print(f'Accuracy: {train_metrics[0]:.4f}, Sensitivity: {train_metrics[1]:.4f}, '
              f'Specificity: {train_metrics[2]:.4f}, AUC: {train_metrics[3]:.4f}')
        print('Validation Metrics:')
        print(f'Accuracy: {val_metrics[0]:.4f}, Sensitivity: {val_metrics[1]:.4f}, '
              f'Specificity: {val_metrics[2]:.4f}, AUC: {val_metrics[3]:.4f}')
        print('-' * 50)
    
    # Print final results
    print('\nFinal Results:')
    metrics_calculator.print_results()
    
    return model

if __name__ == '__main__':
    # Example usage
    model = ClassificationModel()
    # Assuming you have your DataLoaders ready
    # train_loader = DataLoader(...)
    # val_loader = DataLoader(...)
    # trained_model = train_model(model, train_loader, val_loader) 