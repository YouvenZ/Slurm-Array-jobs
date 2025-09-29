import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from parser import return_args_parser_exp, main_parser

args = return_args_parser_exp(parser=main_parser, name='main')

class SimpleNet(nn.Module):
    """Simple CNN for demonstration"""
    def __init__(self, in_channels, num_classes, hidden_dim):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def create_dummy_data(args):
    """Create dummy dataset for demonstration"""
    print("Creating dummy data...")
    
    # Generate random images and labels
    X = np.random.randn(args.data_size, args.in_channels, args.img_height, args.img_width).astype(np.float32)
    y = np.random.randint(0, args.num_classes, args.data_size)
    
    # Split into train/val
    split_idx = int(args.data_size * args.train_split)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Train data shape: {X_train.shape}, Train labels shape: {y_train.shape}")
    print(f"Val data shape: {X_val.shape}, Val labels shape: {y_val.shape}")
    
    return X_train, X_val, y_train, y_val

def numpy_data_processing(X_train, X_val):
    """Demonstrate NumPy data processing"""
    print("\nPerforming NumPy data processing...")
    
    # Normalize data
    mean = np.mean(X_train, axis=(0, 2, 3), keepdims=True)
    std = np.std(X_train, axis=(0, 2, 3), keepdims=True)
    
    X_train_norm = (X_train - mean) / (std + 1e-8)
    X_val_norm = (X_val - mean) / (std + 1e-8)
    
    print(f"Original train data - Mean: {np.mean(X_train):.4f}, Std: {np.std(X_train):.4f}")
    print(f"Normalized train data - Mean: {np.mean(X_train_norm):.4f}, Std: {np.std(X_train_norm):.4f}")
    
    # Additional processing: add noise augmentation to training data
    noise_factor = 0.1
    noise = np.random.normal(0, noise_factor, X_train_norm.shape).astype(np.float32)
    X_train_augmented = X_train_norm + noise
    
    print(f"Applied noise augmentation with factor {noise_factor}")
    
    return X_train_augmented, X_val_norm

def get_device(args):
    """Determine the device to use"""
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    return device

def train_epoch(model, dataloader, criterion, optimizer, device, args):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % args.log_interval == 0:
            print(f'Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    val_loss /= len(dataloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

def main(args):
    print("Starting training with the following configuration:")
    print(f"Task ID: {args.SLURM_ARRAY_TASK_ID}")
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_epochs}")
    print()
    
    # Create dummy data
    X_train, X_val, y_train, y_val = create_dummy_data(args)
    
    # NumPy data processing
    if args.use_numpy_processing:
        X_train, X_val = numpy_data_processing(X_train, X_val)
    
    # Convert to PyTorch tensors and create dataloaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Setup device and model
    device = get_device(args)
    model = SimpleNet(args.in_channels, args.num_classes, args.hidden_dim).to(device)
    
    print(f"\nModel architecture:")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, args)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch [{epoch+1}/{args.num_epochs}] ({epoch_time:.1f}s)')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 50)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if args.save_model:
                torch.save(model.state_dict(), f'best_model_task_{args.SLURM_ARRAY_TASK_ID}.pth')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Final NumPy analysis
    if args.use_numpy_processing:
        print("\nPerforming final NumPy analysis...")
        
        # Get model predictions for analysis
        model.eval()
        with torch.no_grad():
            val_predictions = []
            for data, _ in val_loader:
                data = data.to(device)
                outputs = model(data)
                predictions = torch.softmax(outputs, dim=1).cpu().numpy()
                val_predictions.append(predictions)
        
        val_predictions = np.vstack(val_predictions)
        print(f"Prediction confidence stats:")
        print(f"Mean confidence: {np.mean(np.max(val_predictions, axis=1)):.4f}")
        print(f"Std confidence: {np.std(np.max(val_predictions, axis=1)):.4f}")

if __name__ == "__main__":
    main(args)