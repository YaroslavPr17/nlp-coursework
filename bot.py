import copy
import os
from pathlib import Path
from typing import Dict

import numpy as np
import telebot
import torch.cuda
from telebot import types
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from transformers import pipeline

from datasets_ import DatasetLoader

from src.nlp.classification import Pipeline
from src.nlp.application import get_df_by_person, get_df_by_film_and_person
from src.nlp.info_extraction import get_person_characteristics

bot = telebot.TeleBot('6151372769:AAFFdINtx93_dgK2LVg5FDzq5ZyHyI5GH14', parse_mode='MARKDOWN')
image_path = Path('data', 'images')
if not os.path.exists(image_path):
    os.makedirs(image_path)
ne_dataset = DatasetLoader.load_named_entities_dataset()

qa_model_name = "AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
qa_model = pipeline('question-answering', model=qa_model_name, tokenizer=qa_model_name, device=device)

SENTIMENT_TEXT = 'Хочу узнать тональность отзыва!'
PERSON_TEXT = 'Каков конкретный человек?'
ASK_REVIEW = 'Напишите отзыв о чём-либо'
ASK_NAME = "Введите имя человека и id фильма (при необходимости), разделённые знаком дефис ('-')"


class User:
    def __init__(self, chat_id):
        self.chat_id = chat_id
        self.task = None
        self.request_name = None
        self.request_film_id = None


users: Dict[int, User] = {}

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, "Привет ✌️\n"
                                      "Вы можете классифицировать свой отзыв и найти характеристику актёра или персонажа.")
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    keyboard.add(types.KeyboardButton(SENTIMENT_TEXT))
    keyboard.add(types.KeyboardButton(PERSON_TEXT))
    users[message.chat.id] = User(chat_id=message.chat.id)
    bot.send_message(message.chat.id, 'Что Вы хотите сделать?', reply_markup=keyboard)


@bot.message_handler(func=lambda message: message.text == SENTIMENT_TEXT and message.content_type == 'text')
def sentiment_message_handler(message):
    @bot.message_handler(func=lambda message: message.content_type == 'text')
    def review_handler(message):
        if len(message.text.split()) < 10:
            bot.send_message(message.chat.id, 'Рецензия слишком коротка')
            return
        pred, ret_value = Pipeline('classification', preprocess=False)([message.text], return_confs=True,
                                                                       visualize=True)
        id2label = {0: '😞Негативный😞', 1: '😄Позитивный😄', 2: '😄Позитивный😄'}
        bot.send_message(message.chat.id, ret_value['new_doc'])
        with open(ret_value['graph_path'], 'rb') as graph_file:
            bot.send_photo(message.chat.id, graph_file.read(), caption=id2label[pred[0]])
        os.remove(ret_value['graph_path'])

    if message.chat.id not in users:
        users[message.chat.id] = User(message.chat.id)
    users[message.chat.id].task = 'classification'

    msg = bot.send_message(message.chat.id, ASK_REVIEW)
    bot.register_next_step_handler(msg, review_handler)


@bot.message_handler(func=lambda message: message.text == PERSON_TEXT and message.content_type == 'text')
def person_message_handler(message):
    @bot.message_handler(func=lambda name_id_message: name_id_message.content_type == 'text')
    def name_message_handler(name_id_message):
        ANSWERS_THRESHOLD = 0.18

        name, film_id = None, None
        if '-' in name_id_message.text:
            name, film_id = name_id_message.text.split('-')
            name, film_id = name.strip(), film_id.strip()
            bot.send_message(name_id_message.chat.id, f"Запрос: {name}, id: {film_id}")
        else:
            name = name_id_message.text.strip()
            bot.send_message(name_id_message.chat.id, f"Запрос: {name}")

        if name is None or not name:
            bot.send_message(name_id_message.chat.id, 'Не введено имя персоны. Отмена поиска.')
            return

        if not film_id:
            film_id = None
        else:
            users[name_id_message.chat.id].request_film_id = int(film_id)

        users[name_id_message.chat.id].request_name = name

        try:
            if film_id is not None:
                film_id = int(film_id)
        except ValueError:
            bot.send_message(name_id_message.chat.id, 'Некорректный id фильма')
            return

        scores, answers = get_person_characteristics(ne_dataset, users[name_id_message.chat.id].request_name, qa_model,
                                                     users[name_id_message.chat.id].request_film_id, n_qa_sessions=5)

        if not len(scores):
            bot.send_message(name_id_message.chat.id, 'Нет информации про указанного человека :(')
            return

        clean_answers = np.array(list(map(lambda s: s.strip(' .,:'), answers)))
        best_clean_answers = clean_answers[scores > ANSWERS_THRESHOLD]
        out_str = ''
        for answer in best_clean_answers:
            out_str += f'{telebot.formatting.mbold(str(answer.capitalize()))}\n\n'
        bot.send_message(name_id_message.chat.id, out_str)

    if message.chat.id not in users:
        users[message.chat.id] = User(message.chat.id)
    users[message.chat.id].task = 'person_info'

    msg = bot.send_message(message.chat.id, ASK_NAME)
    bot.register_next_step_handler(msg, name_message_handler)




@bot.message_handler(commands=['help'])
def help_message(message):
    bot.send_message(message.chat.id, "Я бот-анализатор отзывов на фильмы!\n"
                                      "Перейдите в меню и выберите нужное действие!'")






@bot.message_handler(commands=['search_film'])
def search_film_pipeline_message(message):
    @bot.message_handler(content_types=['text'])
    def get_film_title(inner_message):
        film_title = inner_message.text
        # print(dataset[dataset["title"].str.contains(film_title)])
        films_accepted = dataset[dataset["title"].str.contains(film_title, case=False)]
        bot.send_message(message.chat.id,
                         str(films_accepted))

    msg = bot.send_message(message.chat.id, "Введите ключевое слово, встречающееся в фильме")
    bot.register_next_step_handler(msg, get_film_title)


@bot.message_handler(content_types=['text'])  # ['text', 'document', 'audio']
def get_text_messages(message):
    if message.text.lower() == "привет":
        bot.send_message(message.from_user.id, "Привет, чем я могу тебе помочь?")
    elif message.text == 'Покажи все фильмы':
        films = DatasetLoader.load_films_Id_Title_Year_dataset()
        bot.send_message(message.from_user.id, str(films['title'].values[:10]))
    else:
        bot.send_message(message.from_user.id, "Я тебя не понимаю. Напиши /help.")


if __name__ == "__main__":
    print('Bot started!')
    bot.polling(none_stop=True, interval=0)
