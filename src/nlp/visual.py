from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from typing import Iterable, Tuple


def get_n_most_frequent_ngrams(corpus: Iterable[str],
                               ngram_range: Tuple[int, ...] = (1, 1),
                               top_n: int = 10) -> pd.DataFrame:
    print(f'{ngram_range=}')
    print('Building vocabulary...')
    cnt_vec = CountVectorizer(ngram_range=ngram_range)
    cnt_vec.fit(corpus)
    print('Vectorizing documents...')
    docs_embs = cnt_vec.transform(corpus)
    print(f'{docs_embs.shape=}')

    print('Preparing values...')
    token_freqs = np.array(np.sum(docs_embs, axis=0))[0]
    max_idxs = np.argpartition(token_freqs, -top_n)[-top_n:]
    max_freqs = token_freqs[max_idxs]
    most_frequent_tokens = cnt_vec.get_feature_names_out()[max_idxs]

    concated = pd.DataFrame(np.concatenate((max_freqs.reshape(-1, 1), most_frequent_tokens.reshape(-1, 1)), axis=1),
                            columns=['freqs', 'tokens'])

    concated.sort_values(by=['freqs'], inplace=True, ascending=False, ignore_index=True)

    print(end='\n')
    return concated


def visualize_ngram_occurrences(corpus: Iterable[str], ngram_range: Tuple[int] = (1, 1), top_n: int =10):
    assert len(ngram_range) == 2, "Wrong length of ngram_range parameter. Expected 'ngram_range' == 2."

    fig, ax = plt.subplots(ngram_range[1] - ngram_range[0] + 1, 1,
                           figsize=(3 + ngram_range[1] * 2.5, ngram_range[1] * (1.1 + (top_n - 1) / 8)))

    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])

    for i, ngram in enumerate(range(ngram_range[0], ngram_range[1] + 1)):
        df = get_n_most_frequent_ngrams(corpus, ngram_range=(ngram, ngram), top_n=top_n)
        sns.barplot(x=df['freqs'], y=df['tokens'], ax=ax[i])
        ax[i].set_title(f'Top {top_n} Most Frequently Occuring {(ngram, ngram)} ngrams', weight='bold')
        ax[i].grid(color='gainsboro')

    plt.tight_layout()
