from pathlib import Path
import pandas as pd

import dill

from src.kinopoisk_analyzer.utils.constants import datasets_path


def load_top_best_films_ids() -> pd.DataFrame:
    filename = 'top_best_films_Ids.df'

    try:
        with open(Path(datasets_path, filename), 'rb') as file:
            return dill.load(file)
    except FileNotFoundError:
        print(f'No file named {filename} in dataset directory.')


def load_top_popular_films_ids() -> pd.DataFrame:
    filename = 'top_popular_films_Ids.df'

    try:
        with open(Path(datasets_path, filename), 'rb') as file:
            return dill.load(file)
    except FileNotFoundError:
        print(f'No file named {filename} in dataset directory.')


def load_reviews() -> pd.DataFrame:
    filename = 'reviews.df'

    try:
        with open(Path(datasets_path, filename), 'rb') as file:
            return dill.load(file)
    except FileNotFoundError:
        print(f'No file named {filename} in dataset directory.')


def load_reviews_Review_Label() -> pd.DataFrame:
    filename = 'reviews_Review_Label.df'

    try:
        with open(Path(datasets_path, filename), 'rb') as file:
            return dill.load(file)
    except FileNotFoundError:
        print(f'No file named {filename} in dataset directory.')

def load_reviews_Review_Label_clean() -> pd.DataFrame:
    filename = 'reviews_Review_Label_clean.df'

    try:
        with open(Path(datasets_path, filename), 'rb') as file:
            return dill.load(file)
    except FileNotFoundError:
        print(f'No file named {filename} in dataset directory.')


def load_films() -> pd.DataFrame:
    filename = 'films.df'

    try:
        with open(Path(datasets_path, filename), 'rb') as file:
            return dill.load(file)
    except FileNotFoundError:
        print(f'No file named {filename} in dataset directory.')