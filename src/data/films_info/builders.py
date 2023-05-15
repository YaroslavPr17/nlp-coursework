from src.data.films_info.functions import get_genres, get_countries


class FilmParametersBuilder:
    def __init__(self):
        pass

    def set_country(self, country: str):
        if country not in get_countries().keys():
            raise ValueError(f'The value of country ({country}) is incorrect.')
        self.countries = country

    def set_genre(self, genre: str):
        if genre not in get_genres().keys():
            raise ValueError(f'The value of genre ({genre}) is incorrect.')
        self.genres = genre

    def set_order(self, order: str):
        if order not in ['RATING', 'NUM_VOTE', 'YEAR']:
            raise ValueError(f'The value of order ({order}) is incorrect.')
        self.order = order

    def set_type(self, type_: str):
        if type_ not in ['ALL', 'FILM', 'TV_SHOW', 'TV_SERIES', 'MINI_SERIES']:
            raise ValueError(f'The value of type ({type_}) is incorrect.')
        self.type = type_

    def set_rating_from(self, rating_from: int):
        if not 0 <= rating_from <= 10:
            raise ValueError(f'The given rating is {rating_from}. Not in 0..10.')
        self.ratingFrom = rating_from

    def set_rating_to(self, rating_to: int):
        if not 0 <= rating_to <= 10:
            raise ValueError(f'The given rating is {rating_to}. Not in 0..10.')
        self.ratingTo = rating_to

    def set_year_from(self, year_from: int):
        if not isinstance(year_from, int):
            raise TypeError(f'Wrong type of year value. Caught {type(year_from)}, required int.')
        self.yearFrom = year_from

    def set_year_to(self, year_to: int):
        if not isinstance(year_to, int):
            raise TypeError(f'Wrong type of year value. Caught {type(year_to)}, required int.')
        self.yearTo = year_to

    def set_keyword(self, keyword: str):
        self.keyword = keyword

    def build(self):
        return self.__dict__
