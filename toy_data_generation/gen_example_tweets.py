import pandas as pd
import random
from datetime import datetime, timedelta

words_list = open('words_list.txt', 'r').read().split(sep='\n')


def write_sentence(n):
    
    if n > len(words_list):
        return "Requested number of words exceeds the words list length."
    
    selected_words = random.sample(words_list, n)
    
    return ' '.join(selected_words)


def gen_tweets(num_tweets, min_length, max_length):

    tweets_list = []
    for _ in range(num_tweets):
        tweets_list.append(write_sentence(random.randint(min_length, max_length)))
    
    return tweets_list


def generate_random_datetimes(length, min_date, max_date):
    min_datetime = datetime.strptime(min_date, "%Y-%m-%d %H:%M:%S")
    max_datetime = datetime.strptime(max_date, "%Y-%m-%d %H:%M:%S")
    delta = max_datetime - min_datetime
    
    return [min_datetime + timedelta(seconds=random.randint(0, int(delta.total_seconds()))) for _ in range(length)]


def gen_example_tweets(num_rows, min_date, max_date):

    data_dict = dict()
    data_dict["text"] = gen_tweets(num_rows, 3, 7)
    data_dict["datetime"] = generate_random_datetimes(length=num_rows, min_date=min_date, max_date=max_date)

    return pd.DataFrame(data=data_dict)
