from pathlib import Path
from typing import Literal

import pandas as pd

from src.kinopoisk_analyzer.utils.constants import datasets_path


class DatasetLoader:

    @classmethod
    def load_reviews_Review_Label_dataset(cls,
                                          tokenizer: Literal['razdel', 'rutokenizer'] = None,
                                          stopwords: Literal['no', 'nltk'] = 'nltk',
                                          show_path: bool = False) -> pd.DataFrame:
        subfolder = 'reviews_Review_Label'
        if tokenizer:
            filename = f'{subfolder}_{tokenizer}_{stopwords}.csv'
        else:
            filename = f'{subfolder}.csv'

        if show_path:
            print(Path(datasets_path, subfolder, filename))

        try:
            dataset = pd.read_csv(Path(datasets_path, subfolder, filename), index_col=0)
            return dataset
        except FileNotFoundError:
            print(f"No file named '{filename}' in dataset directory.")

    @classmethod
    def load_reviews(cls, show_path: bool = False):
        filename = f'reviews.csv'

        if show_path:
            print(Path(datasets_path, filename))

        try:
            dataset = pd.read_csv(Path(datasets_path, filename), index_col=0)
            return dataset
        except FileNotFoundError:
            print(f"No file named '{filename}' in dataset directory.")

    @classmethod
    def load_top_best_films_ids(cls, show_path: bool = False):
        filename = f'top_best_films_Ids.csv'

        if show_path:
            print(Path(datasets_path, filename))

        try:
            dataset = pd.read_csv(Path(datasets_path, filename), index_col=0)
            return dataset
        except FileNotFoundError:
            print(f"No file named '{filename}' in dataset directory.")

    @classmethod
    def load_top_popular_films_ids(cls, show_path: bool = False):
        filename = f'top_popular_films_Ids.csv'

        if show_path:
            print(Path(datasets_path, filename))

        try:
            dataset = pd.read_csv(Path(datasets_path, filename), index_col=0)
            return dataset
        except FileNotFoundError:
            print(f"No file named '{filename}' in dataset directory.")

    @classmethod
    def load_films(cls, show_path: bool = False):
        filename = f'films.csv'

        if show_path:
            print(Path(datasets_path, filename))

        try:
            dataset = pd.read_csv(Path(datasets_path, filename), index_col=0)
            return dataset
        except FileNotFoundError:
            print(f"No file named '{filename}' in dataset directory.")
