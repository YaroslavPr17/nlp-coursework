import copy
import os
from typing import Iterable

import dill
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from nltk import WhitespaceTokenizer
from sklearn.feature_extraction.text import CountVectorizer


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
        self.ret_value = {}

    def classify(self, docs: Iterable[str], return_confs=False, visualize=False):
        NEGATIVE_THRESHOLD = 0.35

        if self.preprocess:
            # print('Preprocessing...')
            from src.nlp.preprocessing import clean
            if isinstance(docs, pd.Series) and 1 < os.cpu_count() < len(docs):
                docs = docs.parallel_apply(lambda doc: clean(doc, lowercase=True, char_clean=True,
                                                             tokenize=True, tokenizer_type='razdel',
                                                             lemmatize=True, lemmatizer_type='pymorphy2',
                                                             remove_stopwords=False))
            else:
                docs = list(map(lambda doc: clean(doc, lowercase=True, char_clean=True,
                                                  tokenize=True, tokenizer_type='razdel',
                                                  lemmatize=True, lemmatizer_type='pymorphy2',
                                                  remove_stopwords=False),
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

        X_texts = self.vectorizer.transform(docs)

        if visualize and len(docs) == 1:
            REQUIRED_WORDS_LIMIT = 10

            X_texts_markers: np.ndarray = (X_texts.toarray() > 0)[0]
            model_coefs = self.model.coef_[0][X_texts_markers]
            model_features = self.vectorizer.get_feature_names_out()[X_texts_markers]

            WORDS_LIMIT = min(len(model_features) - 1, REQUIRED_WORDS_LIMIT)
            print(WORDS_LIMIT)

            # positive predictions
            positive_coefs = model_coefs[np.argpartition(model_coefs, -WORDS_LIMIT)[-WORDS_LIMIT:]]
            positive_features = model_features[np.argpartition(model_coefs, -WORDS_LIMIT)[-WORDS_LIMIT:]]

            # negative predictions
            negative_coefs = model_coefs[np.argpartition(model_coefs, WORDS_LIMIT)[:WORDS_LIMIT]]
            negative_features = model_features[np.argpartition(model_coefs, WORDS_LIMIT)[:WORDS_LIMIT]]

            all_features = np.append(negative_features, positive_features)
            all_coefs = np.append(negative_coefs, positive_coefs)

            plt.xlabel('Важность слова для модели')
            plt.ylabel('Список слов')

            important_words = all_features[np.abs(all_coefs) > 0.1]

            new_doc = copy.deepcopy(docs[0])
            for word in WhitespaceTokenizer().tokenize(new_doc):
                if word.lower() in important_words:
                    new_doc = new_doc.replace(f' {word} ', f' *{word}* ')

            plt.barh(all_features, all_coefs)

            GRAPH_PATH = 'data/images/graph.png'
            plt.savefig(GRAPH_PATH, dpi=200, bbox_inches='tight')

            self.ret_value['new_doc'] = new_doc
            self.ret_value['graph_path'] = GRAPH_PATH

            plt.clf()

        if return_confs:
            if self.model_type == 'linear':
                self.ret_value['confs'] = self.model.decision_function(X_texts)

        # probas = self.model.predict_proba(X_texts.astype(np.float64))
        # classes = np.where(probas > NEGATIVE_THRESHOLD, 0, 2)[:, 0]

        return self.model.predict(X_texts.astype(np.float64)), self.ret_value

    def __call__(self, docs: Iterable[str], *args, **kwargs):
        if self.task == 'classification':
            return self.classify(docs=docs,
                                 **kwargs)
