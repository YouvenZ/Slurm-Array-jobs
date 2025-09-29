from parser import return_args_parser_exp, main_parser  
import numpy as np
import scipy.optimize
import scipy.stats
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time
import os

args = return_args_parser_exp(parser=main_parser, name='main')

def generate_simple_data(args):
    """Generate simple dummy dataset"""
    print("Generating dummy data...")
    
    np.random.seed(42 + args.SLURM_ARRAY_TASK_ID)
    
    # Smaller dataset for demo
    n_samples = min(args.data_size, 1000)  # Cap at 1000 samples
    n_features = 100  # Fixed feature size for demo
    
    # Generate random data with some structure
    X = np.random.randn(n_samples, n_features)
    
    # Create labels based on simple linear combination
    weights = np.random.randn(n_features)
    scores = X @ weights
    y = (scores > 0).astype(int)  # Binary classification
    
    # Split data
    split_idx = int(n_samples * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Data shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def simple_preprocessing(X_train, X_test):
    """Simple data preprocessing"""
    print("Preprocessing data...")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Simple PCA for dimensionality reduction
    pca = PCA(n_components=20)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"PCA reduced features to: {X_train_pca.shape[1]}")
    return X_train_pca, X_test_pca

def simple_optimization(X_train, y_train):
    """Simple logistic regression using SciPy optimization"""
    print("Optimizing simple model...")
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def loss_function(weights):
        predictions = sigmoid(X_train @ weights)
        # Binary cross-entropy loss
        loss = -np.mean(y_train * np.log(predictions + 1e-8) + 
                       (1 - y_train) * np.log(1 - predictions + 1e-8))
        return loss
    
    # Initialize weights
    initial_weights = np.random.randn(X_train.shape[1]) * 0.01
    
    # Optimize
    result = scipy.optimize.minimize(loss_function, initial_weights, method='BFGS')
    
    print(f"Optimization converged: {result.success}")
    print(f"Final loss: {result.fun:.4f}")
    
    return result.x

def numpy_computations(X_data):
    """Perform various NumPy computations for CPU usage"""
    print("Performing NumPy computations...")
    
    # Matrix operations
    correlation_matrix = np.corrcoef(X_data.T)
    eigenvals, eigenvecs = np.linalg.eig(correlation_matrix)
    
    # Statistical computations
    means = np.mean(X_data, axis=0)
    stds = np.std(X_data, axis=0)
    
    # Matrix multiplication (CPU intensive)
    result_matrix = X_data.T @ X_data
    
    # SVD decomposition
    U, s, Vt = np.linalg.svd(X_data, full_matrices=False)
    
    print(f"Computed eigenvalues range: [{np.min(eigenvals):.3f}, {np.max(eigenvals):.3f}]")
    print(f"Matrix condition number: {np.linalg.cond(correlation_matrix):.2f}")
    
    return {
        'eigenvalues': eigenvals,
        'singular_values': s,
        'condition_number': np.linalg.cond(correlation_matrix)
    }

def main(args):
    print("="*50)
    print("SIMPLE CPU DEMO - NumPy/SciPy Computations")
    print(f"Task ID: {args.SLURM_ARRAY_TASK_ID}")
    print("="*50)
    
    start_time = time.time()
    
    # Generate and preprocess data
    X_train, X_test, y_train, y_test = generate_simple_data(args)
    X_train_processed, X_test_processed = simple_preprocessing(X_train, X_test)
    
    # Optimize model
    optimal_weights = simple_optimization(X_train_processed, y_train)
    
    # Make predictions
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    test_predictions = sigmoid(X_test_processed @ optimal_weights)
    binary_predictions = (test_predictions > 0.5).astype(int)
    accuracy = accuracy_score(y_test, binary_predictions)
    
    # Perform additional NumPy computations
    computation_results = numpy_computations(X_train_processed)
    
    # Some intensive NumPy operations for CPU usage
    print("Performing intensive computations...")
    for i in range(5):
        # Random matrix operations
        large_matrix = np.random.randn(200, 200)
        _ = np.linalg.inv(large_matrix @ large_matrix.T + np.eye(200))
        
        # FFT operations
        signal = np.random.randn(1024)
        _ = np.fft.fft(signal)
        
        print(f"Iteration {i+1}/5 completed")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Task ID: {args.SLURM_ARRAY_TASK_ID}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Processing Time: {total_time:.2f} seconds")
    print(f"Condition Number: {computation_results['condition_number']:.2f}")
    print(f"Top Singular Value: {np.max(computation_results['singular_values']):.3f}")
    print("="*50)
    
    # Save simple results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    results = {
        'task_id': args.SLURM_ARRAY_TASK_ID,
        'accuracy': accuracy,
        'processing_time': total_time,
        'condition_number': computation_results['condition_number']
    }
    
    # Save as text file for simplicity
    with open(f"{results_dir}/simple_results_task_{args.SLURM_ARRAY_TASK_ID}.txt", 'w') as f:
        f.write(f"Task ID: {args.SLURM_ARRAY_TASK_ID}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Processing Time: {total_time:.2f} seconds\n")
        f.write(f"Condition Number: {computation_results['condition_number']:.2f}\n")
    
    print(f"Results saved to results/simple_results_task_{args.SLURM_ARRAY_TASK_ID}.txt")

if __name__ == "__main__":
    main(args)