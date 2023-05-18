import re
from typing import Callable, List


def clean(review: str, lowercase: bool = False, char_clean: bool = False, lemmatize: bool = False,
          tokenize: bool = False, remove_stopwords: bool = False,
          tokenizer_type: str = None, lemmatizer_type: str = 'pymorphy2', stopwords: List[str] = None,
          tagger=None, lemmatizer=None):

    if lowercase:
        review = review.lower()

    if char_clean:
        _html_pattern = r'<[^<]+?>'
        _escape_pattern = r'[\n|\r|\b]'
        review = re.sub(_html_pattern, '  ', review)
        review = re.sub(_escape_pattern, '  ', review)

    if tokenize:
        if tokenizer_type == 'razdel':
            import razdel
            review_tokens = [token.text for token in list(razdel.tokenize(review))]

        elif tokenizer_type == 'TreebankWordTokenizer':
            from nltk.tokenize import TreebankWordTokenizer
            review_tokens = TreebankWordTokenizer().tokenize(review)

        elif tokenizer_type == 'rutokenizer':
            import rutokenizer
            tokenizer = rutokenizer.Tokenizer()
            tokenizer.load()
            review_tokens = tokenizer.tokenize(review)
        else:
            raise NotImplementedError('Unknown tokenizer')

        if lemmatize:
            if lemmatizer_type == 'pymorphy2':
                import pymorphy2
                morph = pymorphy2.MorphAnalyzer(lang='ru')
                review_tokens = [morph.parse(token)[0].normal_form for token in review_tokens]

            elif lemmatizer_type == 'rulemma':
                tags = tagger.tag(review_tokens)
                lemmas = lemmatizer.lemmatize(tags)
                review_tokens = list(map(lambda tpl: tpl[2], lemmas))
            else:
                raise NotImplementedError('Unknown lemmatizer')

        if remove_stopwords:
            review_tokens = [token for token in review_tokens if token not in stopwords]

        review = ' '.join(review_tokens)

    return review
