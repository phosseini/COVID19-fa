import copy
import pandas as pd

from datetime import timedelta
from data_reader import DataLoader
from pre_processing import standardize_tweet_time


# def get_hashtags():
#    hashtags = {"#کرونا": "fa", "#کروناویروس": "fa", "#coronavirus #Iran": "en", "#COVID19 #Iran": "en",
#                "#AyatollahsSpreadCOVID19": "en", "#کرونا_از_آمریکا": "fa", "#در_خانه_بمانیم": "fa",
#                "#کارزار_کرونا": "fa", "کرونا": "fa", "#قرنطینه_خانگی": "fa", "#ویروس_کرونا": "fa"}
#    return hashtags


def get_hashtags():
    hashtags = {"#کرونا": "fa", "#کروناویروس": "fa",
                "#coronavirus #Iran": "en",
                "#COVID19 #Iran": "en",
                "#AyatollahsSpreadCOVID19": "en", "کرونا": "fa", "#ویروس_کرونا": "fa"}
    return hashtags


def get_time_bins(start_date, bin_size, bin_length_day):
    """
    creating a list of time bins
    :param start_date: a string in MM-DD-YYYY format. E.g., 03-12-2020
    :param bin_size: an integer for number of time bins
    :param bin_length_day: period of each time bin in day
    :return: list of lists. E.g., [[s1, e1],[s2, e2]]
    """
    bins = []
    a = pd.to_datetime(start_date)
    for i in range(bin_size):
        b = a + timedelta(days=bin_length_day)
        bins.append([pd.to_datetime(a), pd.to_datetime(b)])
        a = copy.deepcopy(b)
    return bins


def tweets_count_by_day():
    """
    getting count of tweets per day
    :return:
    """

    df = DataLoader().load_tweets(n_count=1000000)

    df['created_at'] = df['created_at'].apply(standardize_tweet_time)
    df['created_at'] = pd.to_datetime(df['created_at'], format='%Y-%m-%d %H:%M:%S')

    tweets_created_at = pd.to_datetime(df['created_at'])
    tweets_counts_df = tweets_created_at.groupby(tweets_created_at.dt.floor('d')).size().reset_index(name='count')

    return tweets_counts_df
