import dill

vocabulary = {}

with open('russian.txt', 'rt', encoding='UTF-8') as rus_countries:
    with open('english.txt', 'rt', encoding='UTF-8') as eng_countries:
        rus_str = rus_countries.readline().strip()
        eng_str = eng_countries.readline().strip()
        while rus_str:
            print(rus_str, eng_str)
            vocabulary[rus_str] = eng_str
            rus_str = rus_countries.readline().strip()
            eng_str = eng_countries.readline().strip()

print(vocabulary)

with open('ru_en_translations.kp', 'wb') as file:
    dill.dump(vocabulary, file)


