import dataclasses
import os
from typing import Dict

import numpy as np
import pandas as pd
import telebot
import torch
from pandarallel import pandarallel
from telebot import types
from transformers import pipeline

from datasets_ import DatasetLoader
from src.nlp.application import get_df_by_person
from src.nlp.classification import Pipeline
from src.nlp.info_extraction import get_person_characteristics, make_film_representation
from src.utils.string_constants import *
from src.utils.constants import image_path


@dataclasses.dataclass
class User:
    def __init__(self, chat_id):
        self.chat_id = chat_id
        self.task = None
        self.person_name_requested = None
        self.film_id_requested = None
        self.df = None


class Env:
    def __init__(self):
        self.users: Dict[int, User] = {}
        self.bot = telebot.TeleBot('6151372769:AAFFdINtx93_dgK2LVg5FDzq5ZyHyI5GH14', parse_mode='MARKDOWN')
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        self.ne_dataset = DatasetLoader.load_named_entities_dataset()
        self.id_title_year_dataset = DatasetLoader.load_films_Id_Title_Year_dataset().rename(columns={'id': 'film_id'})

        self.qa_model_name = "AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru"
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.qa_model = pipeline('question-answering',
                                 model=self.qa_model_name,
                                 tokenizer=self.qa_model_name,
                                 device=self.device)

        pandarallel.initialize()

    def push_user(self, chat_id: int):
        self.users[chat_id] = User(chat_id)

    def set_user_task(self, chat_id: int, task_type: str):
        if chat_id not in self.users:
            self.push_user(chat_id)
        self.users[chat_id].task = task_type

    def set_person_name(self, chat_id: int, person_name: str):
        if chat_id not in self.users:
            self.push_user(chat_id)
        self.users[chat_id].person_name_requested = person_name

    def set_film_id(self, chat_id: int, film_id: int):
        if chat_id not in self.users:
            self.push_user(chat_id)
        self.users[chat_id].film_id_requested = film_id


env = Env()
bot = env.bot
users = env.users


def send_input_request_message(chat_id: int):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    keyboard.add(types.KeyboardButton(TASK_ASK_CLF_SENTIMENT_TEXT))
    keyboard.add(types.KeyboardButton(TASK_ASK_PERS_INFO))

    bot.send_message(chat_id, ASK_ACTION, reply_markup=keyboard)


@bot.message_handler(commands=['start'])
def start_message(message):
    chat_id = message.chat.id
    env.push_user(chat_id)

    bot.send_message(chat_id, HELLO_MESSAGE)

    send_input_request_message(chat_id)


# Management filters block  ###########################


@bot.message_handler(
    func=lambda message: message.text == TASK_ASK_CLF_SENTIMENT_TEXT and message.content_type == 'text')
def sentiment_message_handler(message):
    chat_id = message.chat.id
    env.set_user_task(chat_id, 'classification')

    msg = bot.send_message(chat_id, ASK_REVIEW)
    bot.register_next_step_handler(msg, review_handler)


@bot.message_handler(func=lambda message: message.text == TASK_ASK_PERS_INFO and message.content_type == 'text')
def person_message_handler(message):
    chat_id = message.chat.id
    env.set_user_task(chat_id, 'person_info')

    msg = bot.send_message(chat_id, ASK_NAME)
    bot.register_next_step_handler(msg, name_message_handler)


@bot.message_handler(commands=['help'])
def help_message(message):
    bot.send_message(message.chat.id, HELP_MESSAGE)


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    chat_id = message.chat.id

    bot.send_message(chat_id, UNKNOWN_COMMAND)


# Worker filters block ###########################

@bot.message_handler(func=lambda _message: _message.content_type == 'text')
def review_handler(review_message):
    chat_id = review_message.chat.id

    if len(review_message.text.split()) < 10:
        bot.send_message(chat_id, SHORT_MESSAGE_WARNING)
        send_input_request_message(chat_id)
        return
    pred, ret_value = Pipeline('classification', preprocess=False)([review_message.text], return_confs=True,
                                                                   visualize=True)
    id2label = {0: NEGATIVE_LABEL, 1: POSITIVE_LABEL, 2: POSITIVE_LABEL}
    bot.send_message(chat_id, ret_value['new_doc'])
    with open(ret_value['graph_path'], 'rb') as graph_file:
        bot.send_photo(review_message.chat.id, graph_file.read(), caption=id2label[pred[0]])
    os.remove(ret_value['graph_path'])

    send_input_request_message(chat_id)


@bot.message_handler(func=lambda name_message: name_message.content_type == 'text')
def name_message_handler(name_message):
    chat_id = name_message.chat.id

    name = name_message.text.strip()

    if not name:
        bot.send_message(chat_id, EMPTY_NAME_WARNING)
        return

    env.set_person_name(chat_id, name)

    _df = get_df_by_person(env.ne_dataset, name)
    print(_df.columns)
    if not len(_df):
        bot.send_message(chat_id, NO_FILMS_BY_NAME_WARNING)
        return

    print(_df)
    _df['repr'] = _df['film_id'].parallel_apply(make_film_representation, titles=env.id_title_year_dataset)

    env.users[chat_id].df = _df
    if len(_df['film_id'].unique()) == 1:
        bot.send_message(chat_id, WAITING_FOR_PERSON_INFO_RESPONSE_ANSWER)

        out_string = find_info_and_build_string(env.users[chat_id].df, chat_id)
        bot.send_message(chat_id,
                         out_string if out_string != '' else NO_PROPER_CHARACTERISTICS)

        send_input_request_message(chat_id)
        return

    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    keyboard.add(types.KeyboardButton(SEARCH_ALL_FILMS))
    for rep in np.sort(_df['repr'].unique()):
        keyboard.add(types.KeyboardButton(rep))

    msg = bot.send_message(chat_id, CHOOSE_FILM_OR_REFUSE, reply_markup=keyboard)
    bot.register_next_step_handler(msg, film_title_message_handler)


@bot.message_handler(func=lambda film_title_message: film_title_message.content_type == 'text')
def film_title_message_handler(film_title_message):
    chat_id = film_title_message.chat.id
    film_title = film_title_message.text

    if film_title != SEARCH_ALL_FILMS:
        _df = env.users[chat_id].df
        _df = _df[_df['repr'] == film_title]
        env.users[chat_id].df = _df

    bot.send_message(chat_id, WAITING_FOR_PERSON_INFO_RESPONSE_ANSWER)

    out_string = find_info_and_build_string(env.users[chat_id].df, chat_id)
    bot.send_message(chat_id,
                     out_string if out_string != '' else NO_PROPER_CHARACTERISTICS)

    send_input_request_message(chat_id)


def find_info_and_build_string(ne_data: pd.DataFrame, chat_id: int):
    scores, answers = get_person_characteristics(ne_data, users[chat_id].person_name_requested,
                                                 env.qa_model, users[chat_id].film_id_requested,
                                                 n_qa_sessions=5)
    ANSWERS_THRESHOLD = 0.18
    clean_answers = np.array(list(map(lambda s: s.strip(' .,:'), answers)))
    best_clean_answers = clean_answers[scores > ANSWERS_THRESHOLD]
    out_str = ''
    for answer in best_clean_answers:
        out_str += f'{telebot.formatting.mbold(str(answer.capitalize()))}\n\n'

    return out_str


print('Bot started!')
bot.polling(none_stop=True, interval=0)
