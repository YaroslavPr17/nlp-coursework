from pathlib import Path

import dill
from typing import Dict

from src.kinopoisk_analyzer.utils.constants import data_path


def get_genres() -> Dict[str, int]:
    with open(Path(data_path, 'genres.kp'), 'rb') as genre_file:
        return dill.load(genre_file)


def get_countries() -> Dict[str, int]:
    with open(Path(data_path, 'countries.kp'), 'rb') as country_file:
        return dill.load(country_file)

