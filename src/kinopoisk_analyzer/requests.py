import numpy
import sys

import numpy as np
import requests
import src.kinopoisk_analyzer.utils.constants as constants
from src.kinopoisk_analyzer.utils.stylifiers import Stylers

base_url = 'https://kinopoiskapiunofficial.tech/api/v2.2'


class FilmListRequester:
    def __init__(self, params: dict):
        self.task_url = '/films'
        self.params = params

    def perform(self, page: int) -> requests.Response:
        self.params['page'] = page
        response = requests.get(base_url + self.task_url, params=self.params, headers={'X-API-KEY': constants.X_API_KEY})

        if response.status_code != 200:
            print(f"Response's status_code = {response.status_code}", file=sys.stderr)

        return response


class FiltersRequester:
    def __init__(self):
        self.task_url = '/films/filters'

    def perform(self) -> requests.Response:
        response = requests.get(base_url + self.task_url, headers={'X-API-KEY': constants.X_API_KEY})

        if response.status_code != 200:
            print(f"{Stylers.bold(self.__class__)}: Response's status_code = {response.status_code}", file=sys.stderr)

        return response


class FilmDistributionGenerator:
    def __init__(self, params: dict, per_country: bool = False):
        self.params = params
        self.start_year = params.get('yearFrom')
        self.finish_year = params.get('yearTo')
        if per_country:
            self.country_distribution = {}
        self.linspace = np.linspace(self.start_year, self.finish_year, dtype=numpy.int)
        self.current_year = self.start_year

        print(self.linspace)

    def perform(self, film_data_field: str, consider_nones: bool = False):
        while self.linspace.shape[0] > 1 and self.current_year <= self.linspace[-2]:
            print(self.linspace.shape, end=' ')
            _current_page = 1

            self.params['yearFrom'] = self.current_year
            self.params['yearTo'] = self.linspace[1]

            response = FilmListRequester(self.params).perform(1)

            while response.json().get('items'):
                print('===========================================================')
                print(f'RESPONSE: {self.params.get("yearFrom")}-{self.params.get("yearTo")}')
                print(response.json())

                json_response = response.json().get("items")
                for film in json_response:

                    if film.get("countries") is None:
                        raise IndexError(
                            f'Wrong number of countries for film {film.get("nameRu")} ({film.get("kinopoiskId")}).')

                    if ('yearFrom' in self.params or 'yearTo' in self.params) and film.get('year') is None:
                        print(Stylers.bold(Stylers.red('NOT SAVED')))
                        continue

                    print(Stylers.bold(Stylers.green('SAVED')))

                    if not consider_nones and film.get(film_data_field) is None:
                        break

                    countries = film.get("countries")

                    countries = list(map(lambda lst: lst[0],
                                         list(map(lambda single_cntry_dict: list(single_cntry_dict.values()),
                                                  countries))))

                    print(f'{film.get("nameRu"):40s}: {countries}')
                    film_property = film.get(film_data_field)
                    for country in countries:
                        self.country_distribution.setdefault(country, [])
                        self.country_distribution[country].append(film_property)

                else:
                    _current_page += 1
                    response = FilmListRequester(params=self.params).perform(_current_page)

                    continue

                break

            self.current_year = self.linspace[1]
            self.linspace = np.delete(self.linspace, 0)

        return self.country_distribution
