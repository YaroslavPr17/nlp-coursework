import fastapi
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Union, Optional
import numpy as np
from pydantic import BaseModel

from src.kinopoisk_analyzer.FilmAnalyzer.numvisual import get_visualization
from src.kinopoisk_analyzer.FilmAnalyzer.builders import FilmParametersBuilder

app = FastAPI(debug=True)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/visualizations")
def read_item(prop: str, function: int, consider_nones: bool, source: str,
              countries: Union[str, None] = None,
              keyword: Union[str, None] = None,
              yearTo: Union[int, None] = None,
              yearFrom: Union[int, None] = None,
              ratingTo: Union[int, None] = None,
              ratingFrom: Union[int, None] = None,
              genres: Union[str, None] = None,
              type: Union[str, None] = None,
              order: Union[str, None] = None):
    funcs = [
        len,
        np.mean,
        np.max,
        np.min,
        np.median
    ]

    params = {
        'countries': countries,
        'keyword': keyword,
        'yearTo': yearTo,
        'yearFrom': yearFrom,
        'ratingTo': ratingTo,
        'ratingFrom': ratingFrom,
        'genres': genres,
        'type': type,
        'order': order
    }

    keys_to_delete = []
    for param, value in params.items():
        if value is None:
            keys_to_delete.append(param)
    for del_key in keys_to_delete:
        del params[del_key]

    if source == 'browser':
        visual = get_visualization(properties_=[prop], apply_function=funcs[function], consider_nones=consider_nones,
                                   source=source, params=params)
        return HTMLResponse(content=visual, status_code=200)

    return JSONResponse(content={
            'fun_params': [prop, function, consider_nones, source],
            'search_params': params
        })
