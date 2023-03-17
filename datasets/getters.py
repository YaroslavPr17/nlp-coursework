from pathlib import Path

import dill

from src.kinopoisk_analyzer.utils.constants import datasets_path


def load_top_best_films_ids():
    filename = 'top_best_films_Ids.df'

    try:
        with open(Path(datasets_path, filename), 'rb') as file:
            return dill.load(file)
    except FileNotFoundError:
        print(f'No file named {filename} in dataset directory.')


def load_top_popular_films_ids():
    filename = 'top_popular_films_Ids.df'

    try:
        with open(Path(datasets_path, filename), 'rb') as file:
            return dill.load(file)
    except FileNotFoundError:
        print(f'No file named {filename} in dataset directory.')


def load_reviews():
    filename = 'reviews.df'

    try:
        with open(Path(datasets_path, filename), 'rb') as file:
            return dill.load(file)
    except FileNotFoundError:
        print(f'No file named {filename} in dataset directory.')


def load_reviews_Review_Label():
    filename = 'reviews_Review_Label.df'

    try:
        with open(Path(datasets_path, filename), 'rb') as file:
            return dill.load(file)
    except FileNotFoundError:
        print(f'No file named {filename} in dataset directory.')