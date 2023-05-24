from pathlib import Path
from typing import Dict

import dill

from src.data.films_info.constants import data_path


def get_genres() -> Dict[str, int]:
    with open(Path(data_path, 'genres.data'), 'rb') as genre_file:
        return dill.load(genre_file)


def get_countries() -> Dict[str, int]:
    with open(Path(data_path, 'countries.data'), 'rb') as country_file:
        return dill.load(country_file)
