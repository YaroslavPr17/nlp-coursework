import sys
from pathlib import Path

import dill

from src.kinopoisk_analyzer.requests import FiltersRequester
from src.kinopoisk_analyzer.utils.constants import project_path, data_path
import logging

package_path = Path(project_path, 'src', 'kinopoisk_analyzer', 'service')


def _setup_logger(name, log_file, level=logging.INFO):
    """To set up as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def sync_genres():
    genres_logger = _setup_logger('genres_logger', Path(package_path, 'genres_sync.log'))

    genres_dict = {dct.get('genre'): dct.get('id')
                   for dct
                   in FiltersRequester().perform().json().get('genres')}

    genres_logger.info(f'{len(genres_dict)} genres detected.')

    with open(Path(data_path, 'genres.kp'), 'wb') as genre_file:
        dill.dump(genres_dict, genre_file)


def sync_countries():
    countries_logger = _setup_logger('countries_logger', Path(package_path, 'countries_sync.log'))
    countries_dict = {dct.get('country'): dct.get('id')
                      for dct
                      in FiltersRequester().perform().json().get('countries')}

    countries_logger.info(f'{len(countries_dict)} countries detected.')

    with open(Path(data_path, 'countries.kp'), 'wb') as countries_file:
        dill.dump(countries_dict, countries_file)






