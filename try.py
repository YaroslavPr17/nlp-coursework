import fastapi
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from typing import Union, Optional
import numpy as np
from pydantic import BaseModel

from src.kinopoisk_analyzer.FilmAnalyzer.numvisual import get_visualization
from src.kinopoisk_analyzer.FilmAnalyzer.builders import FilmParametersBuilder

app = FastAPI(debug=True)


class Params(BaseModel):
    country: str
    keyword: str
    year_to: int
    year_from: int
    rating_to: int
    rating_from: int
    genre: str
    type_: str
    order: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/visualizations")
def read_item(prop: Optional[str] = 'ratingKinopoisk',
              function: Optional[int] = 1,
              consider_nones: Optional[int] = 1,
              source: Optional[str] = 'browser',
              params: Params = None):
    print(f'FEATURES: {prop=}, {function=}, {consider_nones=}, {params=}')

    funcs = [
        len,
        np.mean,
        np.max,
        np.min,
        np.median
    ]

    builder = FilmParametersBuilder()
    builder.set_keyword('алиса')

    visual = get_visualization(properties_=['ratingKinopoisk'], apply_function=np.mean, consider_nones=False,
                               source=source, params=builder.build())

    return HTMLResponse(content=visual, status_code=200)
