import pandas as pd

from os import listdir
from os.path import isfile, join

import data_utils as du


class DataLoader:
    def __init__(self):
        self.cleaned_tweet_path = "data/cleaned/"

    def load_tweets(self, n_count, convert_time=False):
        """
        loading cleaned/processed tweets
        :return:
        """
        files = [f for f in listdir(self.cleaned_tweet_path) if isfile(join(self.cleaned_tweet_path, f))]

        df = pd.DataFrame(columns=[])

        for file in files:
            if file.endswith(".xlsx") and not file.startswith("~$"):
                df = df.append(pd.read_excel(self.cleaned_tweet_path + file))

        if convert_time:
            df['created_at'] = pd.to_datetime(df['created_at'])

        # if we do not want to return all records, choose a sample of n_count records
        if n_count < len(df):
            df = df.sample(n_count)

        return df.reset_index()


class DataReader:
    def __init__(self):
        self.data_path = "data/input/"

    def tweet_count(self):
        """
        reading the count of different types of tweets by day
        :return:
        """

        count_dict = {}

        files = [f for f in listdir(self.data_path) if isfile(join(self.data_path, f))]

        filters = {"tweet_type": ['retweet', 'quote', 'reply', 'original'], "lang": ["fa"]}

        hashtags = du.get_hashtags()
        tags = []
        for k, v in hashtags.items():
            tags.append(k.replace('\n', '').replace('\r', '').replace('#', '').strip())

        for file in files:
            if file.endswith(".xlsx") and not file.startswith("~$"):
                df = pd.read_excel(self.data_path + file)

                for index, row in df.iterrows():
                    if row["tweet_type"] in filters["tweet_type"] and row["lang"] in filters["lang"] and any(
                            tag in row["text"] for tag in tags):
                        date = pd.to_datetime(du.standardize_tweet_time(row["created_at"]), format='%Y-%m-%d').date()

                        # check if the day is already in the dictionary
                        if date not in count_dict:
                            count_dict[date] = {"retweet": 0, "quote": 0, "reply": 0, "original": 0}

                        count_dict[date][row["tweet_type"]] += 1

        # reformatting
        count_dict = pd.DataFrame(count_dict).T
        count_dict["date"] = count_dict.index
        count_dict = count_dict.reset_index(drop=True)
        count_dict = count_dict.sort_values(count_dict.columns[4], ascending=True)

        return count_dict

    @staticmethod
    def tweets_count_by_day():
        """
        getting count of tweets per day
        :return:
        """

        df = DataLoader().load_tweets(n_count=10000000)

        df['created_at'] = df['created_at'].apply(du.standardize_tweet_time)
        df['created_at'] = pd.to_datetime(df['created_at'], format='%Y-%m-%d %H:%M:%S')

        tweets_created_at = pd.to_datetime(df['created_at'])
        tweets_counts_df = tweets_created_at.groupby(tweets_created_at.dt.floor('d')).size().reset_index(name='count')

        return tweets_counts_df
