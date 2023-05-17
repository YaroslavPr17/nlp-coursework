import sys
from collections import defaultdict
from typing import List

import dill
import pandas as pd
from natasha import Doc, Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, NewsNERTagger
from tqdm import tqdm


def tag_ne(text, segmenter, morph_tagger, syntax_parser, ner_tagger) -> Doc:
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    doc.tag_ner(ner_tagger)

    return doc


def extract_named_entities(data: pd.DataFrame) -> pd.DataFrame:
    segmenter = Segmenter()
    morph_vocab = MorphVocab()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    syntax_parser = NewsSyntaxParser(emb)
    ner_tagger = NewsNERTagger(emb)

    ne_related_sentences: dict[int, defaultdict[List]] = {}

    nents = pd.DataFrame(columns=[0, 1, 'film_id'])

    film_ids = data['film_id'].unique()

    for i, film_id in enumerate(tqdm(film_ids, ncols=100, file=sys.stdout)):

        dataset = data[data['film_id'] == film_id]
        reviews = dataset['review']
        tagged_reviews = reviews.apply(tag_ne, args=(segmenter, morph_tagger, syntax_parser, ner_tagger))

        for tagged_review in tagged_reviews.values:

            if film_id not in ne_related_sentences:
                ne_related_sentences[film_id] = defaultdict(list)

            for sent in tagged_review.sents:
                for span in sent.spans:
                    if span.type == 'PER':
                        span.normalize(morph_vocab)
                        ne_related_sentences[film_id][span.normal].append(sent.text)
                        break

        new_part = pd.DataFrame(ne_related_sentences[film_id].items())
        new_part['film_id'] = film_id
        nents = pd.concat([nents, new_part], axis=0)

        if i % 100 == 0:
            with open('ne_related_sentences', 'wb') as f:
                dill.dump(nents, f)

    nents.columns = ['ne', 'occurrences', 'film_id']
    nents['n_sents'] = nents['occurrences'].apply(len)

    nents.reset_index().drop(columns='index').to_csv('data/named_entities.csv')

    return nents
