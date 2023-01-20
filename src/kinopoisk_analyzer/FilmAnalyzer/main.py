import ast
import json
import re
import sys
from pathlib import Path
from typing import Callable, Union

import dill
import folium
import numpy as np
import pandas as pd
import requests
from typing.re import Pattern

from src.kinopoisk_analyzer.requests import FilmListRequester
from src.kinopoisk_analyzer.utils.constants import data_path


def get_visualization(property_: str, apply_function: Callable, consider_nones: bool = True, source: str = 'browser'):
    country_distribution = aggregate_films_per_country_by(property_, consider_nones)

    coordinates = make_geographical_coordinates_for_countries(country_distribution.keys())

    stat_data = prepare_data(apply_function, country_distribution)

    # -------------------------------------------------------------------------------------

    folium_map = folium.Map(location=[60, 100], zoom_start=1, width='100%')

    scheme = {
        'type': 'FeatureCollection',
        'features': coordinates
    }

    folium.Choropleth(
        geo_data=scheme,
        name="choropleth",
        data=stat_data,
        columns=["Countries", "Metrics"],
        key_on="feature.properties.country_name",
        fill_color="BuPu",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Number of films released",

    ).add_to(folium_map)

    folium.LayerControl().add_to(folium_map)

    return folium_map._repr_html_() if source == 'browser' else folium_map


def aggregate_films_per_country_by(film_data_field: str, consider_nones=True):
    """
    :param film_data_field: The name of the film property
    :param consider_nones: Whether None values are involved in statistical measurement
    :return: Distribution over countries which produced listed films
    """

    _current_page = 1

    request_params = {'order': 'RATING', 'type': 'ALL',
                      'ratingFrom': 1, 'ratingTo': 10,
                      'yearFrom': 1000, 'yearTo': 3000,
                      # 'keyword': 'алиса'
                      }

    response = FilmListRequester(params=request_params).perform(page=1)

    print(response)

    country_distribution = {}

    while len(response.json().get('items')):
        json_response = response.json().get("items")
        for film in json_response:

            if film.get("countries") is None:
                raise IndexError(
                    f'Wrong number of countries for film {film.get("nameRu")} ({film.get("kinopoiskId")}).')

            if not consider_nones and film.get(film_data_field) is None:
                break

            countries = film.get("countries")

            countries = list(map(lambda lst: lst[0],
                                 list(map(lambda single_cntry_dict: list(single_cntry_dict.values()), countries))))

            print(countries)
            film_property = film.get(film_data_field)
            for country in countries:
                country_distribution.setdefault(country, [])
                country_distribution[country].append(film_property)

        else:
            _current_page += 1
            response = FilmListRequester(params=request_params).perform(_current_page)

            print(f'CURRENT_PAGE = {_current_page}')
            print(f"NEW_RESPONSE = {response.json()}")

            continue

        break

    if 'Германия (ФРГ)' in country_distribution:
        if 'Германия' in country_distribution:
            country_distribution.get('Германия').extend(country_distribution.get('Германия (ФРГ)'))
        else:
            country_distribution['Германия'] = country_distribution.get('Германия (ФРГ)')
        del country_distribution['Германия (ФРГ)']

    if 'СССР' in country_distribution:
        if 'Россия' in country_distribution:
            country_distribution.get('Россия').extend(country_distribution.get('СССР'))
        else:
            country_distribution['Россия'] = country_distribution.get('СССР')
        del country_distribution['СССР']

    print("current_page =", _current_page)
    print("country_distribution =", country_distribution)

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

        print(response, end='\n\n')

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
    print(stat_data)

    return stat_data
