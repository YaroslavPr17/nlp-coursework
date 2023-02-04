from pathlib import Path

import dill
from src.kinopoisk_analyzer.utils.requests import ReviewRequester
from src.kinopoisk_analyzer.utils.stylifiers import Stylers
from src.kinopoisk_analyzer.ReviewAnalyzer.utils.constants import data_path

from typing import Dict


def retrieve_reviews(film_id: int, save: bool = False):
    requester = ReviewRequester()
    response = requester.perform(film_id, {'page': 1})
    json_response: dict = response.json()

    total_pages: int = json_response.get('totalPages')
    print(f'{total_pages=}')

    reviews: Dict[str, dict] = {}
    _current_page: int = 1

    while review_items := json_response.get('items'):
        print(_current_page, end=' ')
        for review in review_items:
            reviews[review.get('description')] = {}
            for field in review:
                if field != 'description':
                    reviews[review.get('description')][field] = review.get(field)
        _current_page += 1
        json_response = requester.perform(film_id, {'page': _current_page}).json()

    with open('logs.txt', 'wt') as f:
        for field in reviews:
            print(Stylers.bold(str(field)), file=f)
            for inner_key, inner_value in reviews[field].items():
                print(f'{inner_key}: {inner_value}', file=f)

            print('\n\n--------------------------------------------------------------------------------------\n\n',
                  file=f)

    if save:
        with open(Path(data_path, 'reviews', f'{str(film_id)}.rv'), 'wb') as reviews_data:
            dill.dump(reviews, reviews_data)

    return reviews
