import sys

import requests
from src.utils import constants
from src.utils.stylifiers import Stylers

BASE_URL_2_2 = 'https://kinopoiskapiunofficial.tech/api/v2.2'
BASE_URL_2_1 = 'https://kinopoiskapiunofficial.tech/api/v2.1'

TIMEOUT_DELAY = 10  # seconds


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
        response = requests.get(BASE_URL_2_2 + self.task_url, params=self.params, timeout=TIMEOUT_DELAY,
                                headers={'X-API-KEY': constants.X_API_KEY})

        if response.status_code != 200:
            Requester.print_error(self.__class__, response.status_code)
            # print(f"Response's status_code = {response.status_code}", file=sys.stderr)

        return response


class FilmRequester(Requester):
    def __init__(self):
        super().__init__()
        self.task_url = '/films'

    def perform(self, film_id: int) -> requests.Response:
        response = requests.get(f"{BASE_URL_2_2}{self.task_url}/{film_id}", timeout=TIMEOUT_DELAY,
                                headers={'X-API-KEY': constants.X_API_KEY})

        if response.status_code != 200:
            Requester.print_error(self.__class__, response.status_code)
            # print(f"Response's status_code = {response.status_code}", file=sys.stderr)

        return response


class FilmListRequesterByKeyword(Requester):
    def __init__(self, keyword: str):
        super().__init__()
        self.task_url = '/films/search-by-keyword'
        self.params = {'keyword': keyword}

    def perform(self, page: int) -> requests.Response:
        self.params['page'] = page
        response = requests.get(BASE_URL_2_1 + self.task_url, params=self.params, timeout=TIMEOUT_DELAY,
                                headers={'X-API-KEY': constants.X_API_KEY})

        if response.status_code != 200:
            Requester.print_error(self.__class__, response.status_code)
            # print(f"Response's status_code = {response.status_code}", file=sys.stderr)

        return response


class FiltersRequester:
    def __init__(self):
        self.task_url = '/films/filters'

    def perform(self) -> requests.Response:
        response = requests.get(BASE_URL_2_2 + self.task_url, timeout=TIMEOUT_DELAY,
                                headers={'X-API-KEY': constants.X_API_KEY})

        if response.status_code != 200:
            Requester.print_error(self.__class__, response.status_code)
            # print(f"{Stylers.bold(self.__class__)}: Response's status_code = {response.status_code}", file=sys.stderr)

        return response


class ReviewRequester:
    def __init__(self):
        self.task_url = '/films/{id}/reviews'

    def perform(self, id_: int, params: dict) -> requests.Response:
        response = requests.get(BASE_URL_2_2 + self.task_url.replace('{id}', str(id_)),
                                params=params, timeout=TIMEOUT_DELAY,
                                headers={'X-API-KEY': constants.X_API_KEY})

        if response.status_code != 200:
            Requester.print_error(self.__class__, response.status_code)

        return response


class TopFilmsRequester:
    top250_best = 'TOP_250_BEST_FILMS'
    top100_popular = 'TOP_100_POPULAR_FILMS'
    top_await = 'TOP_AWAIT_FILMS'

    def __init__(self):
        self.task_url = '/films/top'

    def perform(self, top_type: str, page: int) -> requests.Response:
        response = requests.get(BASE_URL_2_2 + self.task_url,
                                params={'type': top_type, 'page': page}, timeout=TIMEOUT_DELAY,
                                headers={'X-API-KEY': constants.X_API_KEY})

        if response.status_code != 200:
            Requester.print_error(self.__class__, response.status_code)
            # print(f"{Stylers.bold(self.__class__)}: Response's status_code = {response.status_code}", file=sys.stderr)

        return response
