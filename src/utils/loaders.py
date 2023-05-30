import os.path
from pathlib import Path
from typing import Literal, Tuple, Union

import dill
import gdown
import pandas as pd
from sklearn.model_selection import train_test_split as train_test_split_

from src.utils.constants import models_path

links = {
    'logreg_tokrazdel_stopno_100k.model': 'https://drive.google.com/file/d/119DdDcuHN5ASeKsrWvrxlwk7qgwxo2MW/view?usp=share_link',
    'lightgbm_tokrazdel_stopno_100k.model': 'https://drive.google.com/file/d/1j-gah-mj2os8sUDLN4riUhA8V63H_48p/view?usp=share_link',
    'logreg_count_vectorizer_1_4_100000.vocab': 'https://drive.google.com/file/d/17_TpDHup1YhEXZayGvLXziQLUmWCCjAy/view?usp=share_link',
    'lightgbm_count_vectorizer_1_4_100000.vocab': 'https://drive.google.com/file/d/1Is7ONOuMuZ2BIDNZQyvA_S4YqdxDRrsY/view?usp=share_link',
    'logreg_tokrazdel_stopno_500k.model': 'https://drive.google.com/file/d/1KeD3rP4pUkbBZKrCZZXigYVM-fWqiO8H/view?usp=share_link',
    'lightgbm_tokrazdel_stopno_500k.model': 'https://drive.google.com/file/d/1jSubcmaa4grYoO9F15TYLHDagtsuWHE2/view?usp=share_link',
    'count_vectorizer_1_4_500k.vocab': 'https://drive.google.com/file/d/11fpCcKAGGgtZh5mZzQEsMSYBR7FgUmIn/view?usp=share_link',

}


def load_model(filename: str,
               show_path: bool = False,
               subfolder: str = '',
               force_remote: bool = False
               ):

    models_path_ = models_path

    path = Path(models_path_, subfolder, filename)

    if show_path:
        print(path)

    if not os.path.exists(models_path):
        os.makedirs(models_path)
        if subfolder:
            os.makedirs(Path(models_path, subfolder))
    else:
        if not os.path.exists(Path(models_path, subfolder)):
            os.makedirs(Path(models_path, subfolder))

    if os.path.exists(path) and not force_remote:
        print(f"Model '{filename}' was found in local storage.")
    else:
        print(f"Model '{filename}' will be loaded from remote storage.")
        gdown.download(links[filename],
                       output=str(path), fuzzy=True)

    try:
        with open(path, 'rb') as f:
            model = dill.load(f)
        return model
    except FileNotFoundError:
        print(f"No file named '{filename}' in model directory.")
