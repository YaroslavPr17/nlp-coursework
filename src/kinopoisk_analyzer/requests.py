import sys

import requests
import src.kinopoisk_analyzer.utils.constants as constants
from src.kinopoisk_analyzer.utils.stylifiers import Stylers

base_url = 'https://kinopoiskapiunofficial.tech/api/v2.2'


class FilmListRequester:
    def __init__(self, params):
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
