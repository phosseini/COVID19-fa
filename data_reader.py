import pandas as pd

from os import listdir
from os.path import isfile, join


class DataLoader:
    def __init__(self):
        self.cleaned_tweet_path = "data/cleaned/"

    def load_tweets(self):
        """
        loading cleaned/processed tweets
        :return:
        """
        files = [f for f in listdir(self.cleaned_tweet_path) if isfile(join(self.cleaned_tweet_path, f))]

        df = pd.DataFrame(columns=[])

        for file in files:
            if file.endswith(".xlsx") and not file.startswith("~$"):
                df = df.append(pd.read_excel(self.cleaned_tweet_path + file))

        return df.reset_index()
