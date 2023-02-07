import collections
from pathlib import Path
from pprint import pprint
from typing import List
import nltk

import dill

from src.kinopoisk_analyzer.ReviewAnalyzer.utils.constants import data_path


def read_reviews(id_: int) -> List[str]:
    with open(Path(data_path, 'reviews', f'{id_}.rv'), 'rb') as reviews_file:
        reviews = list(dill.load(reviews_file).keys())

    print(type(reviews))

    return reviews


def tokenize_text(tokenizer, text, **tokenizer_kwargs):
    tokens = tokenizer.tokenize(text, **tokenizer_kwargs)
    print(tokens[:3], '...', tokens[-3:], '|', len(tokens), 'tokens.')
    return tokens


def tokenize_corpus(texts, tokenizer, **tokenizer_kwargs):
    return [tokenize_text(tokenizer, text, **tokenizer_kwargs) for text in texts]


def build_vocabulary(tokenized_texts):
    word_counts = collections.defaultdict(int)
    doc_n = 0

    for txt in tokenized_texts:
        doc_n += 1
        unique_text_tokens = set(txt)
        for token in unique_text_tokens:
            word_counts[token] += 1


def test():
    reviews: list = read_reviews(405609)
    token_matrix = tokenize_corpus(reviews, nltk.TreebankWordTokenizer())
    print(len(token_matrix))
    print()
    print(token_matrix[0])





