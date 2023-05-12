import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from datasets_ import DatasetLoader

bot = telebot.TeleBot('6151372769:AAFFdINtx93_dgK2LVg5FDzq5ZyHyI5GH14')
dataset = DatasetLoader.load_films_Id_Title_Year_dataset()


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, "Привет ✌️")
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton('Выбрать фильм', callback_data='option1'))
    bot.send_message(message.chat.id, 'Что Вы хотите сделать?', reply_markup=keyboard)


@bot.message_handler(commands=['help'])
def help_message(message):
    bot.send_message(message.chat.id, "Я бот-анализатор информации о фильмах!\n"
                                      "Ты можешь написать 'привет'")


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
    bot.polling(none_stop=True, interval=0)
