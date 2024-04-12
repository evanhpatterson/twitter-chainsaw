import numpy as np
import pandas as pd
from datetime import datetime
from wordprop.encode_words import one_hot_encode_words
from wordprop.word_indexer import generate_word_dicts


def encode_data(tweet_data: pd.DataFrame, stock_data: pd.DataFrame):

    # create a dict out of the stock data,
    # makes accessing it easier
    stock_data_dict = dict()
    for i, row in stock_data.iterrows():
        stock_data_dict[row["date"]] = row["stock_price"]

    # concatenate tweets
    words_set = set((' '.join(tweet_data["text"])).split(sep=' '))
    unique_words = list(words_set)

    # get input word indices
    input_word_index, _ = generate_word_dicts(unique_words)

    # extract min date
    min_date = min(stock_data['date'])
    min_datetime = datetime.combine(min_date, datetime.min.time())

    # extract max date
    max_date = max(stock_data['date'])
    max_datetime = datetime.combine(max_date, datetime.min.time())

    # pre-filter it for efficiency
    filtered_tweet_data = tweet_data[
        (tweet_data['datetime'] >= min_datetime) &
        (tweet_data['datetime'] <= max_datetime)]
    
    # find the longest tweet, set max length
    max_tweet_length = 0
    for i, row in filtered_tweet_data.iterrows():
        tweet = row['text']
        tlen = len(tweet.split(sep=' '))
        if tlen > max_tweet_length:
            max_tweet_length = tlen
    
    # inputs and outputs
    data_x_pre = []
    data_y = []

    # iterate through dates
    for date, price in stock_data_dict.items():

        # beginning of day
        min_on_day = datetime.combine(date, datetime.min.time())

        # end of day
        max_on_day = datetime.combine(date, datetime.max.time())

        # all tweets on this day
        tweets_on_day = filtered_tweet_data[
            (filtered_tweet_data['datetime'] >= min_on_day) &
            (filtered_tweet_data['datetime'] <= max_on_day)]
        
        # if there are actually tweets on that day
        if len(tweets_on_day) > 0:

            # get the list of all tweets on that day
            tweet_list = tweets_on_day["text"].to_list()
            
            # create a list of encoded tweets for this day
            encoded_tweets = []
            for tweet in tweet_list:

                # one hot encode this list of words
                encoded_tweet = one_hot_encode_words(
                    text_str=tweet,
                    input_word_index=input_word_index,
                    max_size=max_tweet_length)
                encoded_tweets.append(encoded_tweet)
            
            # convert to numpy array
            encoded_tweets = np.array(encoded_tweets, dtype=np.float32)

            # append to list
            data_x_pre.append(encoded_tweets)
            data_y.append(price)
    
    # find the highest number of tweets in a day
    max_num_tweets = 0
    for tweet_list in data_x_pre:
        num_tweets = len(tweet_list)
        if num_tweets > max_num_tweets:
            max_num_tweets = num_tweets
    
    # initialize input data
    data_x = np.zeros(shape=(len(data_x_pre), max_num_tweets, max_tweet_length, len(input_word_index)), dtype=np.float32)

    # set input data using encoded tweets
    for i, x in enumerate(data_x_pre):
        data_x[i, :len(x)] = x
    
    # set output data
    data_y = np.array(data_y, dtype=np.float32)

    # return inputs and outputs
    return data_x, data_y


