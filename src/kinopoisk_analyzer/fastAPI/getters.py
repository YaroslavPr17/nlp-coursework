import fastapi
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Union
import numpy as np

from src.kinopoisk_analyzer.FilmAnalyzer.numvisual import get_visualization
from src.kinopoisk_analyzer.FilmAnalyzer.builders import FilmParametersBuilder

app = FastAPI(debug=True)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/visualizations")
def read_item(prop: str, function: int, consider_nones: bool, source: str = 'browser', **params):
    funcs = [
        len,
        np.mean,
        np.max,
        np.min,
        np.median
    ]

    builder = FilmParametersBuilder()
    builder.set_keyword('алиса')

    visual = get_visualization(properties_=[prop], apply_function=funcs[function], consider_nones=consider_nones,
                               source=source, params=builder.build())

    response = {
        'data': 1,  # HTMLResponse(content=visual, status_code=200),
        'params': 2  # [prop, function, consider_nones, source, params]
    }

    return 1  # JSONResponse(content=response, status_code=100)
