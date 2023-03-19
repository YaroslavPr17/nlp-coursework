import re
from typing import Callable, List


def clean(review: str, tokenizer: Callable, stopwords: List[str]):
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()

    html_pattern = r'<[^<]+?>'
    escape_pattern = r'[\n|\r|\b]'

    lowered_review = review.lower()

    lowered_review = re.sub(html_pattern, '', lowered_review)
    lowered_review = re.sub(escape_pattern, '', lowered_review)

    review_tokens = tokenizer(lowered_review)
    clear_review = [token for token in review_tokens if token not in stopwords]
    clean_review = ' '.join([morph.parse(token)[0].normal_form for token in clear_review])

    return clean_review
