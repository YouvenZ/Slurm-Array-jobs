# Argument parsing logic

from typing import Callable
import pandas as pd
from argparse import Namespace
import ast
import argparse 
from dataclasses import dataclass
import numpy as np
from utils import save_dict 

def main_parser():
    parser = argparse.ArgumentParser(description="Training Script")

    # Model parameters
    parser.add_argument("--model_name", type=str, default="resnet50", help="Backbone architecture")
    parser.add_argument("--in_channels", type=int, default=3, help="Number of input channels")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of output classes")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer type")
    parser.add_argument("--scheduler", type=str, default="StepLR", help="Learning rate scheduler")
    
    # Data parameters
    parser.add_argument("--data_size", type=int, default=1000, help="Size of dummy dataset")
    parser.add_argument("--img_height", type=int, default=32, help="Image height")
    parser.add_argument("--img_width", type=int, default=32, help="Image width")
    parser.add_argument("--train_split", type=float, default=0.8, help="Training data split ratio")
    
    # Hardware and processing
    parser.add_argument("--device", type=str, default="auto", help="Device to use: auto, cpu, cuda")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--use_numpy_processing", type=bool, default=True, help="Enable NumPy data processing")
    
    # Monitoring and logging
    parser.add_argument("--monitor", type=str, default="val_loss", help="Metric to monitor")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--mode", type=str, default="min", help="Monitoring mode: min or max")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    
    # Experiment tracking
    parser.add_argument("--SLURM_ARRAY_TASK_ID", type=int, default=-1, help="Slurm job ID")
    parser.add_argument("--df_name", type=str, default="study_exp_main", help="dataframe name experiment")
    parser.add_argument("--save_model", type=bool, default=True, help="Save trained model")
    parser.add_argument("--experiment_name", type=str, default="demo_experiment", help="Experiment name")

    args = parser.parse_args()
    return args

def return_args_parser_exp(save_dict_ = True, parser:Callable = None, name = 'main'):
    args = parser()
    if save_dict_:
        vars_ = vars(args)
        save_dict(vars_, f'./default_config_dict_{name}')
    if args.SLURM_ARRAY_TASK_ID != -1:
        df = pd.read_csv(f'./{args.df_name}.csv')
        dict_record = df.to_dict('index')[args.SLURM_ARRAY_TASK_ID]    
        dict_record["SLURM_ARRAY_TASK_ID"] = args.SLURM_ARRAY_TASK_ID
        args = Namespace(**dict_record)
        print(args)
    return args

if __name__ == "__main__":
    args = return_args_parser_exp(save_dict_=True, parser=main_parser, name='main')
    print(args)