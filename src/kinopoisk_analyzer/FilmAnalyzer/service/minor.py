from pathlib import Path

import dill

from src.kinopoisk_analyzer.FilmAnalyzer.utils.constants import data_path


def make_translations(is_verbose=False):
    vocabulary = {}

    with open(Path(data_path, '_russian.txt'), 'rt', encoding='UTF-8') as rus_countries:
        with open(Path(data_path, '_english.txt'), 'rt', encoding='UTF-8') as eng_countries:
            rus_str = rus_countries.readline().strip()
            eng_str = eng_countries.readline().strip()
            while rus_str:
                if is_verbose:
                    print(f'{rus_str:20s} -> {eng_str}')
                vocabulary[rus_str] = eng_str
                rus_str = rus_countries.readline().strip()
                eng_str = eng_countries.readline().strip()

    with open(Path(data_path, 'ru_en_translations.kp'), 'wb') as file:
        dill.dump(vocabulary, file)
        print(f'{len(vocabulary)} translations were written successfully!')


