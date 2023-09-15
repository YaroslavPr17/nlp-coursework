from pathlib import Path
from typing import Optional, Union, List
import os
import random
import json

import pandas as pd

from datasets_ import LDFrame
from src.data.datasets_builder import download_reviews_by_ids_list_and_update_dataset
from src.utils.constants import datasets_path


temp_data_dir = Path(datasets_path, 'temp')
config_path = '.current_config'


def extract(name: str, bucket: Optional[str] = None):
    df = LDFrame.load_from_s3(name, bucket)
    current_operation_id: int = random.randint(0, int(10e3))
    current_df_path = Path('./', temp_data_dir, name.replace('.', str(current_operation_id) + '.'))
    config = {
        "id": current_operation_id,
        "df_path": str(current_df_path),
    }
    with open(config_path, 'w') as jsonfile:
        json.dump(config, jsonfile)
        print('Config created')

    df.to_csv(current_df_path)


def transform(ids_list: List[int]):
    with open(config_path, "r") as jsonfile:
        data = json.load(jsonfile)
        print("Config read successful")
    current_df_path = data['df_path']

    download_reviews_by_ids_list_and_update_dataset(ids_list, True, current_df_path)



def load():
    pass
