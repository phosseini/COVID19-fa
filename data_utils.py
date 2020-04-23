import time
import copy
import pandas as pd

from datetime import timedelta

from os import listdir
from os.path import isfile, join


def get_hashtags():
    """
    list of Persian/Iran COVID-19 related hashtags
    :return:
    """
    hashtags = {
        "#Coronavirus": "en",
        "#coronavirus": "en",
        "#COVID19": "en",
        "#COVIDー19": "en",
        "#AyatollahsSpreadCOVID19": "en",
        "#کرونا": "fa",
        "#کورونا": "fa",
        "#کووید۱۹": "fa",
        "#قرنطینه": "fa",
        "#کووید_۱۹": "fa",
        "#کروناویروس": "fa",
        "#کارزار_کرونا": "fa",
        "#قرنطینه_خانگی": "fa",
        "#در_خانه_بمانیم": "fa",
        "#کرونا_از_آمریکا": "fa",
        "#ويروس_چينى": "fa",
        "#کرونا_ویروس": "fa",
        "#ویروس_کورونا": "fa",
        "#کورونا_ویروس": "fa",
        "#کرونا_در_ایران": "fa",
        "#تيك_تاك_سرنگوني": "fa",
        "#کرونا_را_جدی_بگیریم": "fa",
        "#کادر_درمان_را_دریابید": "fa",
        "#کادر_درمان_را_تجهیز_کنید": "fa",
        "#کادر_درمانی_سپر_انسانی_نیست": "fa",
        "#به_داد_کادر_درمان_برسید": "fa",
        "#کرونا_را_شکست_میدهیم": "fa",
        "#تجهیز_کادر_درمانی": "fa",
        "#سپاه_عامل_کرونا": "fa",
        "#کرونا_در_زندان": "fa",
        "#کرونادرایران": "fa",
        "#ویروس_کرونا": "fa",
    }
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
