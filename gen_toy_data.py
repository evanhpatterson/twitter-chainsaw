from gen_example_tweets import gen_example_tweets
from gen_example_stock_data import gen_stock_data

if __name__=="__main__":
    stock_data = gen_stock_data("2019-01-01", "2019-06-01")

    tweet_data = gen_example_tweets(10000, "2019-01-01 00:00:00", "2019-06-01 23:59:59")

    print(stock_data.head(10))
    print(tweet_data.head(10))
