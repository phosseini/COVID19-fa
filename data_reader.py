import pandas as pd

from os import listdir
from os.path import isfile, join


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
