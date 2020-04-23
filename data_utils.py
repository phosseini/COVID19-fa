import time
import copy
import pandas as pd

from datetime import timedelta
from data_loader import DataLoader

from os import listdir
from os.path import isfile, join


def get_hashtags():
    """
    list of Persian/Iran COVID-19 related hashtags
    :return:
    """
    hashtags = {"کرونا_از_آمریکا": "fa",
                "در_خانه_بمانیم": "fa",
                "کارزار_کرونا": "fa",
                "قرنطینه_خانگی": "fa",
                "کرونا_را_جدی_بگیریم": "fa",
                "ويروس_چينى": "fa",
                "کرونا_ویروس": "fa",
                "قرنطینه": "fa",
                "کرونا": "fa",
                "کروناویروس": "fa",
                "ویروس_کرونا": "fa",
                "COVIDー19": "en",
                "coronavirus": "en",
                "Coronavirus": "en",
                "COVID19": "en",
                "AyatollahsSpreadCOVID19": "en"}
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


def standardize_tweet_time(created_at_time):
    """
    converting tweet created_at time to standard datetime format
    :param created_at_time: tweet's created_at field value
    :return:
    """
    return time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(created_at_time, '%a %b %d %H:%M:%S +0000 %Y'))


def save_tweets_ids():
    """
    saving tweets' ids used in our analysis into a text file
    :return:
    """

    cleaned_tweet_path = "data/cleaned/"
    files = [f for f in listdir(cleaned_tweet_path) if isfile(join(cleaned_tweet_path, f))]

    df = pd.DataFrame(columns=["id"])

    for file in files:
        if file.endswith(".xlsx") and not file.startswith("~$"):
            df = df.append(pd.read_excel(cleaned_tweet_path + file, index_col=1))

    df.reset_index()["index"].to_csv(r'data/tweet_ids.txt', header=None, index=None, sep=' ', mode='a')
