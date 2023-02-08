import sys
from pathlib import Path
from typing import Callable, Union, List

import dill
import folium
import numpy as np
import pandas as pd

from src.kinopoisk_analyzer.utils.requests import FilmListRequester
from src.kinopoisk_analyzer.FilmAnalyzer.utils.constants import data_path
from src.kinopoisk_analyzer.FilmAnalyzer.utils.functions import get_genres, get_countries
from src.kinopoisk_analyzer.utils.stylifiers import Stylers


def get_visualization(properties_: List[str], apply_function: Callable, consider_nones: bool = True,
                      source: str = 'browser', params: dict = None):
    if params is None:
        params = {}
    folium_map = folium.Map(location=[60, 100], zoom_start=1, width='100%')

    for property_name_ in properties_:
        country_distribution = aggregate_films_per_country_by(property_name_, consider_nones, **params)
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

    if source == 'browser':
        print('HTML Version for browser')
        return folium_map._repr_html_()

    print('Standard Version for Jupyter')
    return folium_map


def aggregate_films_per_country_by(film_data_field: str,
                                   consider_nones: bool = True,
                                   **params):
    """
    :param film_data_field: The name of the film property
    :param consider_nones: Whether None values are involved in statistical measurement
    :return: Distribution over countries which produced listed films
    """

    genres_list = get_genres()
    countries_list = get_countries()

    request_params = params
    if 'genres' in request_params:
        request_params['genres'] = [genres_list[request_params['genres']]]
    if 'countries' in request_params:
        request_params['countries'] = [countries_list[request_params['countries']]]

    print(Stylers.bold(f'PARAMS TO SEARCH: \n'), request_params, end='\n\n')

    response = FilmListRequester(params=request_params).perform(page=1)

    country_distribution = {}
    _current_page = 1

    while len(response.json().get('items')):
        print('\n\n===========================================================\n')
        print(f'RESPONSE: page={_current_page}')
        print(response.json())

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

    with open(Path(data_path, 'ru_en_translations.kp'), 'rb') as dictionary_file:
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
