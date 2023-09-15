from pathlib import Path
from typing import List, Optional, Union

import dill
import pandas as pd

from src.data.aggregators import \
    aggregate_reviews_by_ids_list, aggregate_films_by_params
from src.utils.constants import datasets_path
from datasets_ import DatasetLoader

films_path = Path(datasets_path, 'films.csv')
reviews_path = Path(datasets_path, 'reviews.csv')


def save_films_by_params(params: dict) -> pd.DataFrame:
    films = aggregate_films_by_params(params)
    print(f'Info about {len(films)} films loaded by parameters specified.')

    try:
        with open(films_path, 'rb') as file:
            old_films = dill.load(file)
    except FileNotFoundError:
        print('No saved films. New dataset created.')
        with open(films_path, 'wb') as file:
            dill.dump(pd.DataFrame(films), file)
    else:
        print('There are saved films. New films-info will be saved into existing dataset.')
        with open(films_path, 'wb') as file:
            dill.dump(
                pd.concat([old_films, pd.DataFrame(films)],
                          ignore_index=True),
                file)

    with open(films_path, 'rb') as file:
        return dill.load(file)


def download_reviews_by_ids_list_and_update_dataset(ids_list: List[int],
                                                    filter_existing_reviews: Optional[bool] = True,
                                                    old_dataset_name: Optional[str] = 'reviews.csv'
                                                    ) -> None:
    old_reviews: Union[pd.DataFrame, None] = DatasetLoader.load_reviews(filename=old_dataset_name, temporary=True)
    assert (old_reviews is not None) or (not filter_existing_reviews), \
        'Cannot filter existing reviews without old dataset'

    downloaded_reviews = download_reviews_by_ids_list(ids_list, filter_existing_reviews, old_reviews)

    if old_reviews is not None:
        print(f"There are saved reviews. ({old_reviews.shape[0]} items). "
              f"New reviews ({len(downloaded_reviews)} items) will be added to existing dataset.")
        new_reviews = merge_update_dataset(old_reviews, downloaded_reviews)
    else:
        print('No saved reviews. New dataset created.')
        new_reviews = downloaded_reviews

    new_reviews.to_csv(old_dataset_name)
    print('Successfully updated.')


def download_reviews_by_ids_list(ids_list: List[int],
                                 filter_existing_reviews: Optional[bool] = True,
                                 old_dataset: Optional[pd.DataFrame] = None) -> pd.DataFrame:

    if filter_existing_reviews:
        assert old_dataset is not None, 'Info about old dataset should be provided'

    n_existing_reviews = []
    if filter_existing_reviews:
        old_reviews = old_dataset

        stats = old_reviews.groupby('film_id').count().review

        for id_ in ids_list:
            n_existing_reviews.append(stats.get(id_, default=0))

    reviews = aggregate_reviews_by_ids_list(ids_list, n_existing_reviews)

    print(f'Info about {len(reviews)} reviews loaded for {len(ids_list)} films.')

    return pd.DataFrame(reviews).transpose().reset_index(names='review').drop_duplicates(subset=['kinopoiskId'])


def merge_update_dataset(df_old: pd.DataFrame, df_new: pd.DataFrame):
    df_shapes = (df_old.shape[1], df_new.shape[1])
    if df_old is None and df_new is None:
        return pd.DataFrame()
    if df_old is None:
        return df_new
    if df_new is None or df_shapes[0] != df_shapes[1]:
        return df_old
    return pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(subset=['kinopoiskId'])


def save_existing_reviews(reviews: dict) -> pd.DataFrame:
    print(f'Info about {len(reviews)} reviews.df given.')

    try:
        with open(reviews_path, 'rb') as file:
            old_reviews = dill.load(file)
    except FileNotFoundError:
        print('No saved reviews. New dataset created.')
        with open(reviews_path, 'wb') as file:
            dill.dump(pd.DataFrame(reviews).transpose(), file)
    else:
        print(f"There are saved reviews. ({old_reviews.shape[0]} items). "
              f"New reviews ({len(reviews)} items) will be saved into existing dataset.")
        with open(reviews_path, 'wb') as file:
            dill.dump(pd.concat([old_reviews, pd.DataFrame(reviews).transpose()]), file)

    with open(reviews_path, 'rb') as file:
        return dill.load(file)


def save_ids_list_for_top_films(aggregated_top: list, file_infix: str) -> pd.DataFrame:
    assert file_infix in ('best', 'popular'), "Wrong type of infix. Available: 'best' or 'popular'."

    filename = f'top_{file_infix}_films_Ids.df'
    print(f'Info about {len(aggregated_top)} top films given.')
    dataset = pd.DataFrame(list(map(lambda film: film.get('filmId'), aggregated_top)), columns=['id'])

    try:
        with open(Path(datasets_path, filename), 'rb') as file:
            old_top = dill.load(file)
    except FileNotFoundError:
        with open(Path(datasets_path, filename), 'wb') as file:
            print(f'No saved tops. New dataset created at {file.name}.')
            dill.dump(dataset, file)
    else:
        print(f'There are saved tops ({len(old_top)} items). New tops will be saved into existing dataset. '
              f'With duplicated values being dropped.')
        with open(Path(datasets_path, filename), 'wb') as file:
            dataset = pd.concat([old_top, dataset], ignore_index=True)
            dataset = dataset.drop_duplicates(subset=['id'])
            dill.dump(dataset, file)

    with open(Path(datasets_path, filename), 'rb') as file:
        return dill.load(file)


def save_Review_Label_dataset_from_full_dataframe(remove_duplicates: bool = False) -> pd.DataFrame:
    raw_dataset_filename = 'reviews.df'
    new_dataset_filename = 'reviews_Review_Label.df'

    try:
        with open(reviews_path, 'rb') as file:
            reviews: pd.DataFrame = dill.load(file)
            print(f"Shape of full '{raw_dataset_filename}' DataFrame = {reviews.shape}.")
    except FileNotFoundError:
        print(f'There is no file named {raw_dataset_filename} in datasets folder.')
        return pd.DataFrame()

    with open(Path(datasets_path, new_dataset_filename), 'wb') as rev_file:
        dataset = pd.DataFrame([reviews.index, reviews.type]).transpose().set_axis(labels=['review', 'label'],
                                                                                   axis=1)
        if remove_duplicates:
            dataset = dataset.drop_duplicates(subset=['review'])

        dataset = dataset.reset_index().drop(columns=['index'])
        print(f"Shape of '{new_dataset_filename}' dataset = {dataset.shape}")
        dill.dump(dataset, rev_file)
        return dataset
