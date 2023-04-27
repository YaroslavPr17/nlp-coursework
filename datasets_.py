from pathlib import Path
from typing import Literal, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split as train_test_split_

from src.kinopoisk_analyzer.utils.constants import datasets_path


class DatasetLoader:

    @classmethod
    def load_reviews_Review_Label_dataset(cls,
                                          tokenizer: Literal['razdel', 'rutokenizer'] = None,
                                          stopwords: Literal['no', 'nltk'] = 'nltk',
                                          train_test_split: bool = False,
                                          test_size: float = 0.3,
                                          remove_neutral_class: bool = False,
                                          classnames_to_int: bool = False,
                                          random_state=42,
                                          show_path: bool = False) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
        subfolder = 'reviews_Review_Label'
        if tokenizer:
            filename = f'{subfolder}_{tokenizer}_{stopwords}.csv'
        else:
            filename = f'{subfolder}.csv'

        if show_path:
            print(Path(datasets_path, subfolder, filename))

        try:
            dataset = pd.read_csv(Path(datasets_path, subfolder, filename), index_col=0)

            if classnames_to_int:
                label_encoding = {
                    'POSITIVE': 2,
                    'NEUTRAL': 1,
                    'NEGATIVE': 0
                }
                dataset.label = dataset.label.apply(lambda label: label_encoding[label])

            if remove_neutral_class:
                if classnames_to_int:
                    neutral_repr = 1
                else:
                    neutral_repr = 'NEUTRAL'
                dataset = dataset[dataset.label != neutral_repr].reset_index().drop(columns=['index'])

            if train_test_split:
                train, test = train_test_split_(dataset, test_size=test_size, random_state=random_state)
                return train, test
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
