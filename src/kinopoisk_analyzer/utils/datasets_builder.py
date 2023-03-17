from pathlib import Path

import dill
import pandas as pd

from src.kinopoisk_analyzer.utils.aggregators import \
    aggregate_films_by_letter, \
    aggregate_reviews_for_n_films, \
    aggregate_reviews_by_ids_list, FilmListAggregator, aggregate_films_by_params
from src.kinopoisk_analyzer.utils.constants import datasets_path

films_path = Path(datasets_path, 'films.df')
reviews_path = Path(datasets_path, 'reviews.df')


def save_films_by_letter():
    films = aggregate_films_by_letter()
    print(f'Info about {len(films)} films loaded by letter.')

    try:
        with open(films_path, 'rb') as file:
            old_films = dill.load(file)
    except FileNotFoundError:
        print('No saved films. New dataset created.')
        with open(films_path, 'wb') as file:
            dill.dump(pd.DataFrame(films), file)
    else:
        print(f'There are saved films ({len(old_films)} items). New films-info will be saved into existing dataset.')
        with open(films_path, 'wb') as file:
            dill.dump(
                pd.concat([old_films, pd.DataFrame(films)],
                          ignore_index=True),
                file)

    with open(films_path, 'rb') as file:
        return dill.load(file)


def save_films_by_params(params: dict):
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


def save_reviews_for_n_films(n=None):
    reviews = aggregate_reviews_for_n_films(n)
    print(f'Info about {len(reviews)} reviews.df loaded for n films.')

    try:
        with open(reviews_path, 'rb') as file:
            old_reviews = dill.load(file)
    except FileNotFoundError:
        print('No saved reviews.df. New dataset created.')
        with open(reviews_path, 'wb') as file:
            dill.dump(pd.DataFrame(reviews), file)
    else:
        print('There are saved reviews.df. New reviews.df will be saved into existing dataset.')
        with open(reviews_path, 'wb') as file:
            dill.dump(pd.concat([old_reviews, pd.DataFrame(reviews).transpose()]), file)

    with open(reviews_path, 'rb') as file:
        return dill.load(file)


def save_reviews_by_ids_list(ids_list: list):
    reviews = aggregate_reviews_by_ids_list(ids_list)
    print(f'Info about {len(reviews)} reviews.df loaded for n films.')

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


def save_existing_reviews(reviews: dict):
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


def save_ids_list_for_top_films(aggregated_top: list, file_infix: str):
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



def save_Review_Label_dataset_from_full_dataframe(remove_duplicates: bool = False):
    raw_dataset_filename = 'reviews.df'
    new_dataset_filename = 'reviews_Review_Label.df'

    try:
        with open(reviews_path, 'rb') as file:
            reviews: pd.DataFrame = dill.load(file)
            print(f"Shape of full '{raw_dataset_filename}' DataFrame = {reviews.shape}.")
    except FileNotFoundError:
        print(f'There is no file named {raw_dataset_filename} in datasets folder.')
        return

    with open(Path(datasets_path, new_dataset_filename), 'wb') as rev_file:
        dataset = pd.DataFrame([reviews.index, reviews.type]).transpose().set_axis(labels=['review', 'label'],
                                                                                   axis=1)
        if remove_duplicates:
            dataset = dataset.drop_duplicates(subset=['review'])
        print(f"Shape of '{new_dataset_filename}' dataset = {dataset.shape}")
        dill.dump(dataset, rev_file)
        return dataset
