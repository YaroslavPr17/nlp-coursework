import os.path
from pathlib import Path
from typing import Literal, Tuple, Union, Optional

import dill
import gdown
import pandas as pd
from sklearn.model_selection import train_test_split as train_test_split_

from src.utils.constants import datasets_path
from src.dataflow.s3 import fetch_dataset

links = {
    'films_Id_Title.csv': 'https://drive.google.com/file/d/1Ks-HIXBJks5x6ZtFlhN3z7v2WhcVaKNr/view?usp=share_link',
    'films_Id_Title_Year.csv': 'https://drive.google.com/file/d/1ZUlF7uk1RVbgUb7j9V3U0_cfcRkaYf_O/view?usp=share_link',
    'named_entities.csv': 'https://drive.google.com/file/d/1hxNFX8XwmGx8HGtl2llGQFdY0yj3Jthh/view?usp=share_link',
    'reviews.csv': 'https://drive.google.com/file/d/1cbdn7YdW6Sc480wEUU2mkInyCi9QA-31/view?usp=share_link',
    'reviews_Review_FilmId.csv': 'https://drive.google.com/file/d/1r0N5fvqI-xOHOs1zBOe-9mu7xm9Bh-K7/view?usp=share_link',
    'top_best_films_Ids.csv': 'https://drive.google.com/file/d/1sz0yv_00jxT8-CRhX8eEFAHvHSw7KJtg/view?usp=share_link',
    'top_popular_films_Ids.csv': 'https://drive.google.com/file/d/1fG5pO2XcoPpO71bGi3CaJrrVTSJKaZTH/view?usp=share_link',
    'reviews_Review_Label.csv': 'https://drive.google.com/file/d/1Cig_GpQ1tjU93rydkqGFie6ss6bNTNR8/view?usp=share_link',
    'reviews_Review_Label_razdel_nltk.csv': 'https://drive.google.com/file/d/1Lsz_66tIggXUbglzL6khjU0CmuHxVXcr/view?usp=share_link',
    'reviews_Review_Label_razdel_no.csv': 'https://drive.google.com/file/d/1Lsz_66tIggXUbglzL6khjU0CmuHxVXcr/view?usp=share_link',
    'reviews_Review_Label_rutokenizer_nltk.csv': 'https://drive.google.com/file/d/1BAdi-lwwZ1zIsU8-V6ec_NPjCDb3AKsL/view?usp=share_link',
    'reviews_Review_Label_rutokenizer_no.csv': 'https://drive.google.com/file/d/1xnuox7V9K2TQc0O6XxYnDYoAkOG4HGNV/view?usp=share_link',
}


class LDFrame(pd.DataFrame):

    @staticmethod
    def parse_str_dataframe(df_str: str):
        import io
        return pd.read_csv(io.StringIO(df_str), index_col=0)

    @staticmethod
    def load_from_s3(name: str, bucket: Optional[str] = None, parse_bytes_to_dataframe: Optional[bool] = True):
        str_csv = fetch_dataset(name, bucket)

        return LDFrame.parse_str_dataframe(str_csv.decode()) if parse_bytes_to_dataframe else \
            str_csv

    def push_to_s3(self, bucket: Optional[str] = None):
        pass


def load_dataset(filename: str,
                 train_test_split: bool = False,
                 test_size: float = 0.3,
                 show_path: bool = False,
                 subfolder: str = '',
                 random_state: int = 42,
                 custom_datasets_path: str = None,
                 force_remote: bool = False
                 ):
    if custom_datasets_path is not None:
        datasets_path_ = custom_datasets_path
    else:
        datasets_path_ = datasets_path

    path = Path(datasets_path_, subfolder, filename)

    if show_path:
        print(path)

    if not os.path.exists(datasets_path):
        os.makedirs(datasets_path)
        if subfolder:
            os.makedirs(Path(datasets_path, subfolder))
    else:
        if not os.path.exists(Path(datasets_path, subfolder)):
            os.makedirs(Path(datasets_path, subfolder))

    if os.path.exists(path) and not force_remote:
        print(f"Dataset '{filename}' was found in local storage.")
    else:
        print(f"Dataset '{filename}' will be loaded from remote storage.")
        gdown.download(links[filename],
                       output=str(path), fuzzy=True)

    try:
        dataset = pd.read_csv(path, index_col=0)
        if train_test_split:
            train, test = train_test_split_(dataset, test_size=test_size, random_state=random_state)
            return train, test
        return dataset
    except FileNotFoundError:
        print(f"No file named '{filename}' in dataset directory.")


class DatasetLoader:

    @classmethod
    def load_reviews_Review_Label_dataset(cls,
                                          tokenizer: Literal['razdel', 'rutokenizer'] = None,
                                          stopwords: Literal['no', 'nltk'] = 'nltk',
                                          train_test_split: bool = False,
                                          test_size: float = 0.3,
                                          remove_neutral_class: bool = False,
                                          classnames_to_int: bool = False,
                                          random_state: int = 42,
                                          show_path: bool = False,
                                          no_marks: bool = False,
                                          int_to_classnames: bool = False,
                                          **kwargs) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        subfolder = 'reviews_Review_Label'
        if tokenizer:
            if no_marks:
                filename = f'{subfolder}_{tokenizer}_{stopwords}_no_marks.csv'
            else:
                filename = f'{subfolder}_{tokenizer}_{stopwords}.csv'
        else:
            if no_marks:
                filename = f'{subfolder}_no_marks.csv'
            else:
                filename = f'{subfolder}.csv'

        try:
            # dataset = pd.read_csv(Path(datasets_path, subfolder, filename), index_col=0)

            dataset = load_dataset(filename, False, test_size, show_path, subfolder, random_state, **kwargs)

            if classnames_to_int:
                label_encoding = {
                    'POSITIVE': 2,
                    'NEUTRAL': 1,
                    'NEGATIVE': 0
                }
                dataset.label = dataset.label.apply(lambda label: label_encoding[label])

            if int_to_classnames:
                label_encoding = {
                    2: 'POSITIVE',
                    1: 'NEUTRAL',
                    0: 'NEGATIVE'
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
    def load_reviews(cls, filename: str = 'reviews.csv', temporary: bool = False, show_path: bool = False, **kwargs):
        subfolder = 'temp' if temporary else ''

        filename = Path(filename).parts[-1]

        try:
            dataset = load_dataset(filename, subfolder=subfolder, show_path=show_path, **kwargs)
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

    @classmethod
    def load_reviews_Review_FilmId_dataset(cls,
                                           train_test_split: bool = False,
                                           test_size: float = 0.3,
                                           show_path: bool = False,
                                           random_state: int = 42,
                                           **kwargs
                                           ):
        filename = f'reviews_Review_FilmId.csv'

        return load_dataset(filename, train_test_split, test_size, show_path, random_state=random_state, **kwargs)

    @classmethod
    def load_films_Id_Title_Year_dataset(cls,
                                         show_path: bool = False,
                                         **kwargs
                                         ):
        filename = f'films_Id_Title_Year.csv'

        return load_dataset(filename, show_path=show_path, **kwargs)

    @classmethod
    def load_named_entities_dataset(cls,
                                    show_path: bool = False,
                                    **kwargs
                                    ):
        filename = f'named_entities.csv'

        dataset = load_dataset(filename, show_path=show_path, **kwargs)

        # Bug: String looks like list, but NOT real list
        if isinstance(dataset['occurrences'].iloc[0], str):
            import ast
            dataset['occurrences'] = dataset['occurrences'].apply(ast.literal_eval)

        return dataset
