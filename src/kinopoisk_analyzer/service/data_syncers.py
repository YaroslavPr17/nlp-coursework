import json
import sys
from pathlib import Path

import dill
import re
import requests
import tqdm

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
        genres_logger.info(f'List of genres was saved to {Path(data_path, "genres.kp")}.')


def sync_countries():
    countries_logger = _setup_logger('countries_logger', Path(package_path, 'countries_sync.log'))
    countries_dict = {dct.get('country'): dct.get('id')
                      for dct
                      in FiltersRequester().perform().json().get('countries')}

    countries_logger.info(f'{len(countries_dict)} countries detected.')

    with open(Path(data_path, 'countries.kp'), 'wb') as countries_file:
        dill.dump(countries_dict, countries_file)
        countries_logger.info(f'List of countries was saved to {Path(data_path, "countries.kp")}.')


def sync_countries_coordinates():
    coordinates_logger = _setup_logger('coordinates_logger', Path(package_path, 'coordinates_sync.log'))

    github_page = 'https://github.com/georgique/world-geojson/tree/main/countries'
    response = requests.get(github_page)
    pattern = re.compile(r'/\w*.json')
    countries_files_names = pattern.findall(str(response.content))

    _tqdm = tqdm.tqdm(countries_files_names, file=sys.stdout)
    _tqdm.set_description(f'Downloading countries\' border-coordinates')

    n_retrieved = 0
    n_saved = 0
    for country_file_name in _tqdm:

        # Sample country_file_name='/argentina.json'
        _country_file_name = re.split('/', country_file_name)[1]
        data: requests.Response = requests.get(f'https://raw.githubusercontent.com/georgique/world-geojson/main/countries/{_country_file_name}')

        if data.status_code != 200:
            coordinates_logger.error(f'Error in retrieving coordinates information in {_country_file_name}. Error code = {data.status_code}')
            continue

        n_retrieved += 1
        data: dict = data.json()

        # print(data)

        with open(Path(data_path, 'countries_coordinates', _country_file_name), 'wb') as countries_coordinates_file:
            dill.dump(data, countries_coordinates_file)
            n_saved += 1

    coordinates_logger.info(f'Retrieved: {n_retrieved} | Saved: {n_saved} countries\' coordinates.')
