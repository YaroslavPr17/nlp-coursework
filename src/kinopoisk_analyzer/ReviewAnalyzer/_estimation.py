import dill
from src.kinopoisk_analyzer.utils.requests import ReviewRequester
from src.kinopoisk_analyzer.utils.stylifiers import Stylers


def hashing_text():
    requester = ReviewRequester()
    response = requester.perform(405609, {'page': 1})

    vocabulary = {}

    _current_page = 1

    while response_dict := response.json().get('items'):
        print(_current_page, end=' ')
        for review in response_dict:
            vocabulary[review.get('description')] = {}
            for key in review:
                if key != 'description':
                    vocabulary[review.get('description')][key] = review.get(key)
        _current_page += 1
        response = requester.perform(405609, {'page': _current_page})

    with open('logs.txt', 'wt') as f:
        for key in vocabulary:
            print(Stylers.bold(str(key)), file=f)
            for inner_key, inner_value in vocabulary[key].items():
                print(f'{inner_key}: {inner_value}', file=f)

            print('\n\n--------------------------------------------------------------------------------------\n\n', file=f)

    with open('hashing_text_dict.data', 'wb') as f:
        dill.dump(vocabulary, f)


def as_structure():
    requester = ReviewRequester()
    response = requester.perform(405609, {'page': 1})

    vocabulary = {}

    _current_page = 1

    while response_dict := response.json().get('items'):
        print(_current_page, end=' ')
        for review in response_dict:
            vocabulary[review.get('kinopoiskId')] = {}
            for key in review:
                if key != 'kinopoiskId':
                    vocabulary[review.get('kinopoiskId')][key] = review.get(key)
        _current_page += 1
        response = requester.perform(405609, {'page': _current_page})

    with open('logs_as_structure.txt', 'wt') as f:
        for key in vocabulary:
            print(Stylers.bold(str(key)), file=f)
            for inner_key, inner_value in vocabulary[key].items():
                print(f'{inner_key}: {inner_value}', file=f)

            print('\n\n--------------------------------------------------------------------------------------\n\n',
                  file=f)

    with open('as_structure_dict.data', 'wb') as f:
        dill.dump(vocabulary, f)


