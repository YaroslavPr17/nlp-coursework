import sys
from abc import abstractmethod

import requests
import src.kinopoisk_analyzer.utils.constants as constants
from src.kinopoisk_analyzer.utils.stylifiers import Stylers

base_url = 'https://kinopoiskapiunofficial.tech/api/v2.2'


class Requester:
    def __init__(self):
        self.task_url: str
        self.params: dict
    @staticmethod
    def print_error(class_name, status_code: int):
        print(f"{Stylers.bold(class_name)}: Response's status_code = {status_code}", file=sys.stderr)

class FilmListRequester(Requester):
    def __init__(self, params: dict):
        super().__init__()
        self.task_url = '/films'
        self.params = params

    def perform(self, page: int) -> requests.Response:
        self.params['page'] = page
        response = requests.get(base_url + self.task_url, params=self.params,
                                headers={'X-API-KEY': constants.X_API_KEY})

        if response.status_code != 200:
            Requester.print_error(self.__class__, response.status_code)
            # print(f"Response's status_code = {response.status_code}", file=sys.stderr)

        return response


class FiltersRequester:
    def __init__(self):
        self.task_url = '/films/filters'

    def perform(self) -> requests.Response:
        response = requests.get(base_url + self.task_url, headers={'X-API-KEY': constants.X_API_KEY})

        if response.status_code != 200:
            Requester.print_error(self.__class__, response.status_code)
            # print(f"{Stylers.bold(self.__class__)}: Response's status_code = {response.status_code}", file=sys.stderr)

        return response


class ReviewRequester:
    def __init__(self):
        self.task_url = '/films/{id}/reviews'

    def perform(self, id_: int, params: dict) -> requests.Response:
        response = requests.get(base_url + self.task_url.replace('{id}', str(id_)),
                                params=params,
                                headers={'X-API-KEY': constants.X_API_KEY})

        if response.status_code != 200:
            Requester.print_error(self.__class__, response.status_code)
            # print(f"{Stylers.bold(self.__class__)}: Response's status_code = {response.status_code}", file=sys.stderr)

        return response
