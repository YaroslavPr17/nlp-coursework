import sys
from typing import List
import os

import pandas as pd

import numpy as np
from gensim.models import Word2Vec
import fasttext


def tokenize_by_whitespaces(corpus: pd.Series):
    from nltk import WhitespaceTokenizer
    return corpus.apply(WhitespaceTokenizer().tokenize)


def doc2vec(document: List[str], model, emb_size: int = 300):
    sum_doc_meaning = np.zeros(emb_size)
    vectorized_tokens = []

    for token in document:
        try:
            if isinstance(model, Word2Vec):
                sum_doc_meaning += model.wv[token]
            else:
                sum_doc_meaning += model[token]
            vectorized_tokens.append(token)
        except KeyError:
            continue

    return sum_doc_meaning / len(vectorized_tokens)


def corp2vecs(corpus: pd.Series,
              model_type: str,
              emb_size=None,
              pretrained_model=None,
              train=True,
              n_epochs: int = 5,
              n_treads: int = 12):

    _model_types = ('word2vec', 'fasttext')

    if (model_type := model_type.lower()) not in _model_types:
        print(f'Invalid model type. Expected {_model_types}. Exit.', file=sys.stderr)
        return np.array([])

    if pretrained_model is None and not train:
        print('Seems that no trained model will be used for vectorization! Exit.', file=sys.stderr)
        return np.array([])

    if emb_size is None:
        emb_size = 300 if model_type == 'word2vec' else 100

    print('Turning documents to lists of tokens...')
    tokenized_corpus = tokenize_by_whitespaces(corpus)

    if pretrained_model is None:
        print(f'Training NEW {model_type} model on given documents...')
        if model_type == 'word2vec':
            model = Word2Vec(sentences=tokenized_corpus, vector_size=emb_size, workers=n_treads, epochs=n_epochs)
        elif model_type == 'fasttext':
            training_filepath = 'tmp_files/_train_all_reviews.tmp'
            with open(training_filepath, 'wt', encoding='utf-8') as file:
                file.write('         '.join(corpus))
            model = fasttext.train_unsupervised(training_filepath, model='skipgram', dim=emb_size, thread=n_treads,
                                                epoch=n_epochs)
            os.remove(training_filepath)
        else:
            raise NotImplementedError
    else:
        if train:
            print(f'Training PRETRAINED {model_type} model on given documents...')
            raise NotImplementedError
        else:
            print(f'Provided with pretrained {model_type} model. Training skipped.')
            model = pretrained_model

    print('Vectorizing each document...')
    vectorized = tokenized_corpus.apply(doc2vec, args=(model, emb_size)).to_numpy()
    print('Vectorization finished!')

    return model, np.array([*vectorized])
