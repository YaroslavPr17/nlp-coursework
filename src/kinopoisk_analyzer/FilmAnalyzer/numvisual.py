import ast
import json
import re
import sys
from pathlib import Path
from typing import Callable, Union, List

import dill
import folium
import numpy as np
import pandas as pd
from pprint import pprint
import requests
from typing.re import Pattern

from src.kinopoisk_analyzer.requests import FilmDistributionGenerator
from src.kinopoisk_analyzer.utils.constants import data_path
from src.kinopoisk_analyzer.utils.functions import get_genres
from src.kinopoisk_analyzer.utils.stylifiers import Stylers


def get_visualization(properties_: List[str], apply_function: Callable, consider_nones: bool = True,
                      source: str = 'browser'):
    folium_map = folium.Map(location=[60, 100], zoom_start=1, width='100%')

    for property_name_ in properties_:
        country_distribution = aggregate_films_per_country_by(property_name_, consider_nones, keyword='танец с саблями')
        coordinates = make_geographical_coordinates_for_countries(country_distribution.keys())
        stat_data = prepare_data(apply_function, country_distribution)

        scheme = {
            'type': 'FeatureCollection',
            'features': coordinates
        }

        folium.Choropleth(
            geo_data=scheme,
            name=property_name_,
            data=stat_data,
            columns=["Countries", "Metrics"],
            key_on="feature.properties.country_name",
            fill_color="BuPu",
            fill_opacity=0.7,
            line_opacity=0.2,
            bins=10,
            legend_name=f'{apply_function.__name__} function applied for values of {property_name_} grouped for every country.',

        ).add_to(folium_map)

    # -------------------------------------------------------------------------------------

    folium.LayerControl().add_to(folium_map)

    return folium_map._repr_html_() if source == 'browser' else folium_map


def aggregate_films_per_country_by(film_data_field: str,
                                   consider_nones=True,
                                   type_: str = 'ALL',
                                   genre: str = None,
                                   rating_from: int = 1,
                                   rating_to: int = 10,
                                   year_from: int = 1000,
                                   year_to: int = 3000,
                                   keyword: str = None):
    """
    :param keyword:
    :param year_to:
    :param year_from:
    :param rating_to:
    :param rating_from:
    :param genre:
    :param type_:
    :param film_data_field: The name of the film property
    :param consider_nones: Whether None values are involved in statistical measurement
    :return: Distribution over countries which produced listed films
    """

    genres_list = get_genres()
    if genre not in (*genres_list, None):
        raise ValueError(f'The value of genre ({genre}) is incorrect.')
    if type_ not in ['ALL', 'FILM', 'TV_SHOW', 'TV_SERIES', 'MINI_SERIES', None]:
        raise ValueError(f'The value of type ({type_}) is incorrect.')

    request_params = {'order': 'RATING',
                      'type': type_,
                      'ratingFrom': rating_from,
                      'ratingTo': rating_to,
                      'yearFrom': year_from,
                      'yearTo': year_to}
    if genre is not None:
        request_params['genres'] = [genres_list[genre]]
    if keyword is not None:
        request_params['keyword'] = keyword

    country_distribution = FilmDistributionGenerator(params=request_params, per_country=True) \
        .perform(film_data_field, consider_nones)

    # print("current_page =", _current_page)
    print("COUNTRY_DISTRIBUTION =", country_distribution)

    return country_distribution


def make_geographical_coordinates_for_countries(countries):
    def _get_country_borders_info(country_filename: str) -> Union[dict, None]:
        country_info_path = Path(data_path, 'countries_coordinates', country_filename)
        try:
            with open(country_info_path, 'rb') as country_file:
                return dill.load(country_file)
        except FileNotFoundError:
            print(f"Cannot find coordinates of country={country_filename} in storage.", file=sys.stderr)
            return None

    coordinates = []

    _replacements = {
        'ussr': 'russia',
        'united_states': 'usa'
    }

    with open('src/translations/ru_en_translations.kp', 'rb') as dictionary_file:
        dictionary = dill.load(dictionary_file)

    for country in countries:
        if country not in dictionary:
            raise AttributeError(f'No country in translations list: {country}')
        translated_country = dictionary.get(country).lower().replace(' ', '_')
        translated_country = _replacements.get(translated_country, translated_country)

        print(country, translated_country)

        if country == 'Чехословакия':
            czech_response = _get_country_borders_info('czech.json')
            slovakia_response = _get_country_borders_info('slovakia.json')
            response = czech_response
            response.get('features').extend(slovakia_response.get('features'))
        else:
            response = _get_country_borders_info(f'{translated_country}.json')
            if response is None:
                continue

        features = response.get('features')

        for feature in features:
            feature.get('properties')['country_name'] = country
            if country == 'Россия':
                _russia_coordinates_list_ptr = feature.get('geometry').get('coordinates')[0]
                feature.get('geometry').get('coordinates')[0] = list(
                    map(lambda lst: [lst[0] + 360, lst[1]] if lst[0] < 0 else lst,
                        _russia_coordinates_list_ptr)
                )

        coordinates.extend(features)

    return coordinates


def prepare_data(func: Callable, country_distribution):
    df_dict = {}

    for country in country_distribution:
        print(country_distribution[country])
        country_data = np.array(country_distribution[country])
        try:
            df_dict[country] = func(country_data)
        except TypeError:
            print('TypeError caught:\n'
                  f'country_distribution[{country}] = {country_data}',
                  file=sys.stderr)
            df_dict[country] = func(country_data[np.logical_not(pd.isna(country_data))])

    print(df_dict)

    stat_data = pd.DataFrame(data={'Countries': df_dict.keys(), 'Metrics': df_dict.values()})

    return stat_data
