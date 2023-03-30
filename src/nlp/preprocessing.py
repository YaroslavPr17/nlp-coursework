import re
from typing import Callable, List


def clean(review: str, tokenizer_type: str, stopwords: List[str],
          lemmatizer_type: str = 'pymorphy2', tagger=None, lemmatizer=None):
    html_pattern = r'<[^<]+?>'
    escape_pattern = r'[\n|\r|\b]'

    lowered_review = review.lower()

    lowered_review = re.sub(html_pattern, '', lowered_review)
    lowered_review = re.sub(escape_pattern, '', lowered_review)

    if tokenizer_type == 'razdel':
        import razdel
        review_tokens = [token.text for token in list(razdel.tokenize(review))]

    elif tokenizer_type == 'TreebankWordTokenizer':
        from nltk.tokenize import TreebankWordTokenizer
        review_tokens = TreebankWordTokenizer().tokenize(lowered_review)

    elif tokenizer_type == 'rutokenizer':
        import rutokenizer
        tokenizer = rutokenizer.Tokenizer()
        tokenizer.load()
        review_tokens = tokenizer.tokenize(lowered_review)
    else:
        raise NotImplementedError('Unknown tokenizer')

    if lemmatizer_type == 'pymorphy2':
        import pymorphy2
        morph = pymorphy2.MorphAnalyzer(lang='ru')
        clear_review = [morph.parse(token)[0].normal_form for token in review_tokens]

    elif lemmatizer_type == 'rulemma':
        tags = tagger.tag(review_tokens)
        lemmas = lemmatizer.lemmatize(tags)
        clear_review = list(map(lambda tpl: tpl[2], lemmas))
    else:
        raise NotImplementedError('Unknown lemmatizer')

    clean_review = [token for token in clear_review if token not in stopwords]
    final_string = ' '.join(clean_review)

    return final_string
