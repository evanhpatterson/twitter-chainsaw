import numpy as np
import pandas as pd
import pytz
from datetime import timedelta


def get_unique_words(tweet_data: pd.DataFrame, text_col: str):
    '''
        Generate a list of all unique words from the tweets dataset.
    '''
    # get a list of all words
    word_list = tweet_data[text_col]

    # filter elements that are strings
    filtered_word_list = [word for word in word_list if isinstance(word, str)]
    
    # concatenate the words with spaces
    concatenated_words = (' '.join(filtered_word_list)).split()

    # get all unique words
    words_set = set(concatenated_words)

    # turn the set of unique words into a list
    unique_words = list(words_set)

    return unique_words


def generate_word_dicts(words_list: list):
    '''
        Word dicts are generated that map words to indices and indices to words.
    '''
    word_index = dict()
    index_word = dict()
    for i, word in enumerate(words_list):
        word_index[word] = i
        index_word[i] = word
    return word_index, index_word


def convert_df_datetime_to_date(data: pd.DataFrame, datetime_col: str, text_col: str, date_col: str):

    data[datetime_col] = pd.to_datetime(data[datetime_col], errors='coerce')

    # filter out invalid dates
    data = data.dropna(subset=[datetime_col])

    # convert eastern time
    eastern = pytz.timezone('US/Eastern')

    new_df = pd.DataFrame()

    # get date
    new_df[date_col] = data[datetime_col].dt.tz_convert(eastern).dt.date
    new_df[text_col] = data[text_col]
    
    return new_df


def load_dataframes(tweet_fpath: str, stock_fpath: str, tweet_datetime_col: str, text_col: str, date_col: str):
    tweet_data = pd.read_csv(tweet_fpath)
    tweet_data = convert_df_datetime_to_date(
        data=tweet_data,
        datetime_col=tweet_datetime_col,
        text_col=text_col,
        date_col=date_col)

    stock_data = pd.read_csv(stock_fpath, parse_dates=[date_col])

    stock_data[date_col] = stock_data[date_col].dt.date
    
    return tweet_data, stock_data


def get_tweet_dict(tweet_data: pd.DataFrame, text_col: str, date_col: str):

    tweet_dict = dict()

    for tdate in set(tweet_data[date_col].to_list()):
        tweet_dict[tdate] = []

    for i, row in tweet_data.iterrows():
        tweet_dict[row[date_col]].append(row[text_col])
    
    return tweet_dict


def get_stock_dict(stock_data: pd.DataFrame, date_col: str):

    stock_dict = dict()

    for _, row in stock_data.iterrows():
        stock_date = row[date_col]
        stock_row = row.drop(date_col).to_numpy()
        stock_dict[stock_date] = stock_row
    
    return stock_dict


def encode_tweet_list(tweet_list: list, input_word_index: dict, max_tweets: int, max_words: int):
    encoded_tweets = np.zeros((max_tweets, max_words), dtype=int)
    for i, tweet in enumerate(tweet_list):
        words_list = tweet.split()

        filtered_words_list = [word for word in words_list if isinstance(word, str)]

        filtered_words_list = filtered_words_list[:max_words]

        for j, word in enumerate(filtered_words_list):
            if word in input_word_index:
                encoded_tweets[i, j] = input_word_index[word]
    
    return encoded_tweets


def encode_tweet(tweet, input_word_index: dict, max_words: int):
    words_list = tweet.split()
    filtered_words_list = [word for word in words_list if isinstance(word, str)]
    
    if len(filtered_words_list) > max_words:
        filtered_words_list = filtered_words_list[:max_words]
    
    encoded_words = np.zeros(max_words, dtype=int)
    
    for i, word in enumerate(filtered_words_list):
        encoded_words[i] = input_word_index[word]
    
    return encoded_words


def encode_data(
        tweet_dict: dict,
        stock_dict: dict,
        input_word_index: dict,
        max_words: int,
        time_window: int):

    dates = []

    # get dates that are in both
    for date_, _ in tweet_dict.items():
        if date_ in stock_dict:
            dates.append(date_)
    
    data_x = []
    data_y = []

    # iterate through dates
    for cur_date in dates:

        tweets = []

        # go through the days within the time window
        for i in range(time_window):
            td = timedelta(days=i)

            # current day minus i days
            tday = cur_date - td

            if tday in tweet_dict:
                tweets += tweet_dict[tday]
        
        # add if there are actually tweets
        if len(tweets) > 0:

            # encode tweets as ints
            encoded_tweets = [encode_tweet(tweet, input_word_index=input_word_index, max_words=max_words) for tweet in tweets]

            # encode stocks as a numpy array for that day
            encoded_stocks = stock_dict[cur_date]

            # add to lists of data
            data_x.append(encoded_tweets)
            data_y.append(encoded_stocks)
    
    data_y = np.array(data_y, dtype=np.float32)
    
    return data_x, data_y


def preprocess_data(tweet_fpath: str, stock_fpath: str):

    tweets_text_col = 'processed_text'

    # load tweet data and stock data
    tweet_data, stock_data = load_dataframes(
        tweet_fpath=tweet_fpath,
        stock_fpath=stock_fpath,
        tweet_datetime_col='Datetime',
        text_col=tweets_text_col,
        date_col='Date')
    
    # get all unique words from tweets
    unique_words = get_unique_words(tweet_data=tweet_data, text_col=tweets_text_col)
    input_word_index, _ = generate_word_dicts(words_list=unique_words)

    # load tweet dict; the format is date: [tweet1, tweet2, ...]
    tweet_dict = get_tweet_dict(tweet_data=tweet_data, text_col=tweets_text_col, date_col='Date')

    # load stock dict; the format is date: stock_values
    stock_dict = get_stock_dict(stock_data=stock_data, date_col='Date')

    # encode the data numerically
    data_x, data_y = encode_data(
        tweet_dict=tweet_dict,
        stock_dict=stock_dict,
        input_word_index=input_word_index,
        max_words=64,
        time_window=3)
    
    return data_x, data_y, input_word_index, unique_words


if __name__=="__main__":

    tweet_fpath = "data/final_preprocessed_tweets.csv"

    stock_fpath = "data/NVDA-delta.csv"

    data_x, data_y, input_word_index, unique_words = preprocess_data(tweet_fpath=tweet_fpath, stock_fpath=stock_fpath)
