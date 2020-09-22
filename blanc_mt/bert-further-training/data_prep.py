from utils import *
import pandas as pd

df = prepare_data("corpus.en_ru.1m.en", "corpus.en_ru.1m.ru")

df = apply_filters(df, 'russian', 'english')

df.to_csv('yandex_parallel.csv')
