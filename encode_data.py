import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from wordprop.encode_words import convert_to_ints
from wordprop.word_indexer import generate_word_dicts

def keep_overlapping_data(tweet_data: pd.DataFrame, stock_data: pd.DataFrame):
    '''
        Take both dataframes, remove any rows that are not overlapping.
    '''

    # extract min date
    min_datetime_stocks = datetime.combine(min(stock_data['Date']), datetime.min.time())

    # extract max date
    max_datetime_stocks = datetime.combine(max(stock_data['Date']), datetime.min.time())

    # extract min and max tweet dates
    min_datetime_tweets = min(tweet_data['datetime']).to_pydatetime()
    max_datetime_tweets = max(tweet_data['datetime']).to_pydatetime()

    # use the latest minimum and the earliest maximum
    min_datetime = max(min_datetime_tweets, min_datetime_stocks)
    max_datetime = min(max_datetime_tweets, max_datetime_stocks)

    # remove tweet data outside of the dates of interest
    filtered_tweet_data = tweet_data[
        (tweet_data['datetime'] >= min_datetime) &
        (tweet_data['datetime'] <= max_datetime)]
    
    # remove stock data outside of the dates of interest
    filtered_stock_data = stock_data[
        (stock_data['Date'] >= min_datetime.date()) &
        (stock_data['Date'] <= max_datetime.date())]
    
    return filtered_tweet_data, filtered_stock_data
    

def encode_data(tweet_data: pd.DataFrame, stock_data: pd.DataFrame, input_word_index: dict, time_window: int=0):
    '''
        Encode the tweet data as data_x, and the stock data as data_y
    '''

    # make sure there are tweets for every stock and stocks for every tweet
    tweet_data, stock_data = keep_overlapping_data(tweet_data=tweet_data, stock_data=stock_data)

    # create a dict out of the stock data,
    # makes accessing it easier
    stock_data_dict = dict()
    for i, row in stock_data.iterrows():
        row_vals = row.drop("Date")
        stock_data_dict[row["Date"]] = row_vals

    # find the longest tweet, set max length
    max_tweet_length = 0
    for i, row in tweet_data.iterrows():
        tweet = row['text']
        tlen = len(tweet.split())
        if tlen > max_tweet_length:
            max_tweet_length = tlen
    
    # inputs and outputs
    data_x_pre = []
    data_y = []

    min_date_stocks = min(stock_data['Date']) + timedelta(days=time_window)

    stock_data = stock_data[stock_data['Date'] >= min_date_stocks]

    # iterate through dates
    for date, price in stock_data_dict.items():

        # beginning of first day
        min_day_one = datetime.combine(date, datetime.min.time()) - timedelta(days=time_window)

        # end of last day
        max_last_day = datetime.combine(date, datetime.max.time()) - timedelta(days=1)

        # all tweets in this time window
        input_tweets = tweet_data[
            (tweet_data['datetime'] >= min_day_one) &
            (tweet_data['datetime'] <= max_last_day)]
        
        # if there are actually tweets in this time window
        if len(input_tweets) > 0:

            # get the list of all tweets in this window
            tweet_list = input_tweets["text"].to_list()
            
            # create a list of encoded tweets for this window
            encoded_tweets = []
            for tweet in tweet_list:

                # encode this list of words
                encoded_tweet = convert_to_ints(
                    text_str=tweet,
                    input_word_index=input_word_index,
                    max_size=max_tweet_length)
                
                encoded_tweets.append(encoded_tweet)
            
            # convert to numpy array
            encoded_tweets = np.array(encoded_tweets, dtype=int)

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
    data_x = np.zeros(shape=(len(data_x_pre), max_num_tweets, max_tweet_length), dtype=int)

    # set input data using encoded tweets
    for i, x in enumerate(data_x_pre):
        data_x[i, :len(x)] = x
    
    # set output data
    data_y = np.array(data_y, dtype=np.float32)

    # return inputs and outputs
    return data_x, data_y
