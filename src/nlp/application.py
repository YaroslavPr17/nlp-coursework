import sys
from typing import Iterable, List
import dill
import os

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import WhitespaceTokenizer
from tqdm.notebook import tqdm


class Pipeline:
    def __init__(self, task: str,
                 model=None,
                 preprocess=False,
                 model_type: str = 'linear',

                 ):
        import pandarallel
        pandarallel.pandarallel.initialize(nb_workers=os.cpu_count(), verbose=0)

        _task_types = ['classification', 'summarization']

        assert task in ['classification', 'summarization'], f'Wrong type of task. Expected: {_task_types}'

        self.task = task
        self.model = model
        self.preprocess = preprocess
        self.model_type = model_type
        self.vectorizer = None
        self.ret_value = None

    def classify(self, docs: Iterable[str], return_confs=False):
        NEGATIVE_THRESHOLD = 0.35

        if self.preprocess:
            # print('Preprocessing...')
            from src.nlp.preprocessing import clean
            if isinstance(docs, pd.Series) and os.cpu_count() > 1:
                docs = docs.parallel_apply(lambda doc: clean(doc, tokenizer_type='razdel', stopwords=[]))
            else:
                docs = list(map(lambda doc: clean(doc, tokenizer_type='razdel', stopwords=[]),
                                docs))

        # print('Vectorization...')

        if self.model_type == 'linear' and self.model is None:
            with open('models/logreg_tokrazdel_stopno_100k.model', 'rb') as f:
                self.model = dill.load(f)

            with open('models/count_vectorizer_1_4_100000.vocab', 'rb') as f:
                vocab = dill.load(f)
                self.vectorizer = CountVectorizer(tokenizer=WhitespaceTokenizer().tokenize,
                                                  vocabulary=vocab,
                                                  ngram_range=(1, 4))

        if self.model_type == 'lightgbm' and self.model is None:
            with open('models/lightgbm_tokrazdel_stopno_100k.model', 'rb') as f:
                self.model = dill.load(f)

            with open('models/lightgbm_count_vectorizer_1_4_100000.vocab', 'rb') as f:
                vocab = dill.load(f)
                self.vectorizer = CountVectorizer(tokenizer=WhitespaceTokenizer().tokenize,
                                                  vocabulary=vocab,
                                                  ngram_range=(1, 4))

        # print('Classification...')
        X_texts = self.vectorizer.transform(docs)

        if return_confs:
            if self.model_type == 'linear':
                self.ret_value = self.model.decision_function(X_texts)

        # probas = self.model.predict_proba(X_texts.astype(np.float64))
        # classes = np.where(probas > NEGATIVE_THRESHOLD, 0, 2)[:, 0]

        return self.model.predict(X_texts.astype(np.float64)), self.ret_value

    def __call__(self, docs: Iterable[str], return_confs=False, *args, **kwargs):
        if self.task == 'classification':
            return self.classify(docs=docs,
                                 return_confs=return_confs)


def get_df_by_person(data: pd.DataFrame, name: str) -> pd.DataFrame:
    filtered = data[data['ne'].str.contains(name, case=False)].sort_values(by='n_sents', ascending=False)
    return filtered


def get_df_by_film_and_person(data: pd.DataFrame, film_id: int, name: str) -> pd.DataFrame:
    df_with_person = get_df_by_person(data, name)
    filtered = df_with_person[df_with_person['film_id'] == film_id].sort_values(by='n_sents', ascending=False)
    return filtered


def collect_sents_to_summarize(data: pd.DataFrame, n_sents: int = 100) -> List[str]:
    all_sents = np.array(data['occurrences'].sum())
    pl = Pipeline('classification', model_type='linear', preprocess=True)
    _, confs = pl(pd.Series(all_sents), return_confs=True)

    to_summarize = []
    to_summarize.extend(all_sents[confs < 0].tolist())
    _n = n_sents - len(to_summarize)
    _n = _n if _n < len(confs) else len(confs) - len(to_summarize)
    to_summarize.extend(all_sents[np.argpartition(confs, -_n)[-_n:]].tolist())

    return to_summarize


def split_opinions_to_chunks(tokenizer,
                             data: pd.DataFrame = None,
                             sentences: List[str] = None,
                             model_max_input: int = 600,
                             show_info: bool = False):
    if sentences is not None:
        all_sents = sentences
    elif data is not None:
        all_sents = list(data['occurrences'].sum())
    else:
        raise ValueError('No data is provided!')

    chunks = ['']
    chunks_length = [0]

    for sent in all_sents:
        _new_length = chunks_length[-1] + len(tokenizer.encode(sent))
        if _new_length > model_max_input:
            if len(tokenizer.encode(sent)) > model_max_input:
                print('The sentence has length greater that max model input, thus will be skipped', file=sys.stderr)
                continue
            chunks.append('')
            chunks_length.append(0)
            _new_length = chunks_length[-1] + len(tokenizer.encode(sent))

        chunks_length[-1] = _new_length
        chunks[-1] += ' ' + sent

    if show_info:
        import matplotlib.pyplot as plt

        print('Overall number of sentences:', len(all_sents))
        print('Number of chunks:', len(chunks))

        plt.figure(figsize=(10, 3))

        plt.plot(chunks_length)
        plt.title('The number of tokens across all chunks')
        plt.xlabel('Chunk number')
        plt.ylabel('Number of tokens')

        plt.show()

    return chunks
