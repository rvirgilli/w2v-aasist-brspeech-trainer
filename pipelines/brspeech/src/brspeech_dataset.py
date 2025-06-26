from pathlib import Path
import pandas as pd
from typing import Literal

from src.datasets.base_dataset import BaseDataset
from configuration.df_train_config import DF_Train_Config


class BrSpeechDataset(BaseDataset):
    """
    Dataset class for BrSpeech metadata CSVs.
    Inherits from BaseDataset to reuse audio loading and processing logic.
    """
    def __init__(self, config: DF_Train_Config, subset: Literal['train', 'val', 'test']):
        
        # Load metadata CSV
        csv_path = Path(config.root_dir) / f"{subset}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Metadata file not found at: {csv_path}")
        
        samples_df = pd.read_csv(csv_path)

        # Initialize parent class
        super().__init__(
            samples_df=samples_df,
            config=config,
            subset=subset
        )

    def __len__(self) -> int:
        return len(self.samples_df) 