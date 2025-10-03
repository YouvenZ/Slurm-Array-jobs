# SLURM Array Tutorial: Running Parameterized Experiments on a Cluster

Welcome! This repository demonstrates **how to run parameterized experiments on a cluster using SLURM arrays**. You'll learn how to structure your code, generate experiment configurations, and launch jobs efficiently—ideal for machine learning, data science, or any batch computation.

---

## Table of Contents

- [SLURM Array Tutorial: Running Parameterized Experiments on a Cluster](#slurm-array-tutorial-running-parameterized-experiments-on-a-cluster)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Best Practices](#best-practices)
  - [How It Works](#how-it-works)
  - [Step-by-Step Usage](#step-by-step-usage)
    - [1. Argument Parsing](#1-argument-parsing)
    - [2. Generating Experiment Configurations](#2-generating-experiment-configurations)
    - [3. Running Locally vs. on Cluster](#3-running-locally-vs-on-cluster)
    - [4. SLURM Array Job Scripts](#4-slurm-array-job-scripts)
  - [Example Scripts](#example-scripts)
  - [Tips \& Troubleshooting](#tips--troubleshooting)
  - [Credits](#credits)

---

## Overview

**SLURM arrays** allow you to run many jobs in parallel, each with different parameters. This is perfect for hyperparameter sweeps or large-scale experiments.

**Key idea:**  
- Store experiment parameters in a CSV file.
- Each SLURM array job reads one row (using its array index) and runs with those parameters.

---

## Best Practices

- **Parameterize everything:** Use argument parsers for all experiment settings.
- **Save default configs:** Store your parser's defaults for reproducibility.
- **Generate CSVs programmatically:** Avoid manual editing—use scripts!
- **Keep code modular:** Separate parsing, experiment creation, and job logic.
- **Log outputs and errors:** Use SLURM's logging features for debugging.
- **Version your experiment configs:** Keep CSVs and scripts under version control.

---

## How It Works

1. **Argument Parser:**  
   Define all experiment parameters in a Python parser.

2. **CSV Generation:**  
   Create a CSV where each row is a unique experiment configuration.

3. **SLURM Array Job:**  
   Each job reads its row from the CSV (using `$SLURM_ARRAY_TASK_ID`) and runs with those parameters.

---

## Step-by-Step Usage

### 1. Argument Parsing

Define your experiment parameters in `main_parser()`:

```python
import argparse

def main_parser():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    # ... add more parameters ...
    parser.add_argument("--SLURM_ARRAY_TASK_ID", type=int, default=-1)
    parser.add_argument("--df_name", type=str, default="study_exp_main")
    return parser.parse_args()
```

### 2. Generating Experiment Configurations

Use the `ExperimentCreator` class to generate a CSV of experiment settings:

```python
from parser import main_parser, return_args_parser_exp
from utils import load_pickle_dict

class ExperimentCreator:
    # ... see code for full implementation ...
    def create_study_experiment(self, config, output_path='./study_exp_main.csv'):
        # Validates and saves experiment configs to CSV
        pass

def main():
    num_samples = 100
    config = {
        'learning_rate': [0.01 * i for i in range(1, num_samples + 1)],
        'batch_size': [2**(2+i) for i in range(1, num_samples + 1)],
    }
    experiment_creator = ExperimentCreator(num_samples=num_samples, parser_type='main')
    experiment_creator.create_study_experiment(config=config)

if __name__ == "__main__":
    main()
```

### 3. Running Locally vs. on Cluster

Use a wrapper to load parameters from the CSV if running on the cluster:

```python
def return_args_parser_exp(save_dict_=True, parser=None, name='main'):
    args = parser()
    if save_dict_:
        save_dict(vars(args), f'./default_config_dict_{name}')
    if args.SLURM_ARRAY_TASK_ID != -1:
        df = pd.read_csv(f'./{args.df_name}.csv')
        dict_record = df.to_dict('index')[args.SLURM_ARRAY_TASK_ID]
        dict_record["SLURM_ARRAY_TASK_ID"] = args.SLURM_ARRAY_TASK_ID
        args = Namespace(**dict_record)
    return args
```

### 4. SLURM Array Job Scripts

**CPU Example:**

```bash
#!/bin/bash
#SBATCH -p Serveurs-CPU
#SBATCH -J cpu_array_test
#SBATCH -c 4
#SBATCH --mem 8000
#SBATCH --error log/cpu_array_test%A_%a.txt
#SBATCH --output log/cpu_array_test%A_%a.out
#SBATCH --array=[0-49]%25

srun singularity run /path/to/image.sif python /path/to/main_cpu.py --SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID --df_name study_exp_main
echo "Job $SLURM_JOB_ID with array ID $SLURM_ARRAY_TASK_ID has completed."
```

**GPU Example:**

```bash
#!/bin/bash
#SBATCH -p GPU48Go
#SBATCH --gres=gpu
#SBATCH -J gpu_array_test
#SBATCH -c 4
#SBATCH --mem 8000
#SBATCH --error log/gpu_array_test%A_%a.txt
#SBATCH --output log/gpu_array_test%A_%a.out
#SBATCH --array=[0-50]%5

srun singularity run --nv /path/to/image.sif python /path/to/main_gpu.py --SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID --df_name study_exp_main
echo "Job $SLURM_JOB_ID with array ID $SLURM_ARRAY_TASK_ID has completed."
```

**Note:**  
- Change the singularity image and script paths to match your cluster setup.
- Adjust SLURM parameters (`-p`, `--mem`, etc.) as needed.

---

## Example Scripts

- `main_cpu.py`: Example CPU computation using parsed arguments.
- `main_gpu.py`: Example GPU computation using parsed arguments.
- `sbatch_job_create.sh`: Script to generate the experiment CSV.

---

## Tips & Troubleshooting

- **Check your CSV:** Make sure the number of rows matches your SLURM array range.
- **Log everything:** Use SLURM's `--output` and `--error` flags.
- **Test locally:** Run with `--SLURM_ARRAY_TASK_ID -1` before submitting to the cluster.
- **Singularity images:** Ensure your Python environment and dependencies are included.
- **Cluster-specific settings:** Always adapt SLURM flags to your cluster's requirements.

---

## Credits

Created for a YouTube tutorial by [your channel name].  
Feel free to fork, adapt, and share!

---

**Happy experimenting! 🚀**
