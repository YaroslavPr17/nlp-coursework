import fastapi
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from typing import Union
import numpy as np

from src.kinopoisk_analyzer.FilmAnalyzer.main import get_visualization

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/visualizations")
def read_item(prop: str = 'ratingKinopoisk', function: int = 0, consider_nones: int = 1, source: str = 'browser'):
    funcs = [
        len,
        np.mean,
        np.max,
        np.min,
        np.median
    ]
    return HTMLResponse(content=get_visualization(prop, funcs[function], bool(consider_nones), source), status_code=200)

