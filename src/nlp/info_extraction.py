import os
from typing import Tuple

import numpy as np
import pandas as pd
from transformers import QuestionAnsweringPipeline

from src.nlp.application import collect_sents_to_summarize, get_df_by_film_and_person, get_df_by_person


def get_person_characteristics(data: pd.DataFrame,
                               name: str,
                               model: QuestionAnsweringPipeline,
                               film_id: int = None,
                               n_qa_sessions: int = 10,
                               ) -> Tuple[np.ndarray, np.ndarray]:
    from src.nlp.preprocessing import clean
    from random import shuffle

    _N_SENTS = 100

    print('Collecting sentences to retrieve characteristics...')

    if film_id is not None:
        _df = get_df_by_film_and_person(data, film_id, name)
        if not len(_df):
            return np.array([]), np.array([])
    else:
        _df = get_df_by_person(data, name)
        if not len(_df):
            return np.array([]), np.array([])

    listed_opinions = collect_sents_to_summarize(_df, n_sents=_N_SENTS)
    print(f'Собрано {len(listed_opinions)} предложений про именованную сущность')

    shuffle(listed_opinions)

    opinions = '\n'.join(listed_opinions)
    #     print(opinions, end='\n------------------------------------\n')
    opinions = clean(opinions, char_clean=True)
    #     print(opinions)

    print('Built opinions text!')

    answers = []
    scores = []
    questions = [
        {
            'question': f'Какая {name}?',
            'context': None
        },
        {
            'question': f'Что сделал {name}?',
            'context': None
        },
        {
            'question': f'Что хорошо у {name}?',
            'context': None
        },
        {
            'question': f'Что плохо у {name}?',
            'context': None
        }
    ]

    print('Questions initialized!')

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    questions_to_remove = []

    current_session_passed = 0

    try:
        while questions and current_session_passed < n_qa_sessions:

            for i, question in enumerate(questions):
                question['context'] = opinions
                print('Thinking on question number', i, '...')
                answer = model(question)
                print(question['question'])
                print(answer['answer'])
                opinions = opinions.replace(answer['answer'], ' ')
                if answer['answer'] == '':
                    print('question:', question)
                    print(answer['answer'])
                    questions_to_remove.append(i)
                answers.append(answer['answer'])
                scores.append(answer['score'])

            if questions_to_remove:
                print('OLD SET OF QUESTIONS:', questions)
                new_questions = []
                for i, question in enumerate(questions):
                    print('WILL BE EXCLUDED:', question)
                    if i not in questions_to_remove:
                        new_questions.append(question)

                questions = new_questions
                questions_to_remove = []
                print('NEW SET OF QUESTIONS:', questions)

            current_session_passed += 1

    except KeyboardInterrupt:
        return np.array(scores), np.array(answers)

    return np.array(scores), np.array(answers)


def make_film_representation(film_id: int, titles: pd.DataFrame):
    obj = titles[titles['film_id'] == film_id]
    return f"{obj['title'].values[0]} ({obj['year'].values[0]})"
