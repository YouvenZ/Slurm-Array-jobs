import pickle
import os
from typing import Dict, List, Any, Optional
import pandas as pd
from utils import load_pickle_dict
from parser import main_parser, return_args_parser_exp


class ExperimentCreator:
    """Creates experiment configurations and saves them as CSV files."""
    
    def __init__(self, num_samples: int, parser_type: str) -> None:
        self.num_samples = num_samples
        self.parser_type = parser_type
        self.default_config = self._load_default_config()
        self.cwd = os.getcwd()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration dictionary."""
        config_path = f'./default_config_dict_{self.parser_type}'
        return load_pickle_dict(config_path)
    
    def _validate_config(self, config: Dict[str, List]) -> None:
        """Validate experiment configuration parameters."""
        if not config:
            return
            
        for param_name, values in config.items():
            if not isinstance(values, list):
                raise ValueError(f"Parameter '{param_name}' must be a list")
            if len(values) != self.num_samples:
                raise ValueError(
                    f"Parameter '{param_name}' has {len(values)} values, "
                    f"expected {self.num_samples}"
                )
    
    def _create_experiment_rows(self, config: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate experiment configuration rows."""
        if not config:
            return [self.default_config.copy() for _ in range(self.num_samples)]
        
        experiments = []
        parameters = list(config.keys())
        
        for i in range(self.num_samples):
            experiment = self.default_config.copy()
            for param in parameters:
                experiment[param] = config[param][i]
            experiments.append(experiment)
        
        return experiments
    
    def create_study_experiment(self, config: Dict[str, List], 
                              output_path: str = './study_exp_main.csv') -> None:
        """Create experiment study and save to CSV file."""
        self._validate_config(config)
        
        experiments = self._create_experiment_rows(config)
        df_experiments = pd.DataFrame(experiments)
        df_experiments.to_csv(output_path, index=False)
        
        print(f"Experiment file created: {output_path}")


def main() -> None:
    """Main execution function."""


    num_samples = 100 # Number of experiment configurations to create

    config = {
        'learning_rate': [float(0.01*float(i)) for i in range(1, num_samples + 1)],
        'batch_size': [2**(2+i) for i in range(1, num_samples + 1)],
    }


    
    # config = {}
    
    
    args_main = return_args_parser_exp(parser=main_parser, name='main')
    
    experiment_creator = ExperimentCreator(num_samples=num_samples, parser_type='main')
    experiment_creator.create_study_experiment(config=config)


if __name__ == "__main__":
    main()