from __future__ import unicode_literals

import re
import json
import emoji
import string
import pandas as pd

import data_utils as du

from hazm import *
from os import listdir
from os.path import isfile, join

normalizer = Normalizer()
stemmer = Stemmer()
lemmatizer = Lemmatizer()


class PreProcessing:
    def __init__(self):
        self.data_path = "../data/input/"
        self.cleaned_tweet_path = "../data/cleaned/tweets_"

    def clean_data_excel(self, count=100, save_checkpoint=100):
        """
        reading tweets' excel data
        :param count: number of records to be in the final data frame
        :param save_checkpoint: saving point of tweets' data frame
        :return:
        """

        df_columns = ["id", "tweet_url", "created_at", "text", "user_description", "user_followers_count",
                      "user_friends_count", "user_location", "user_statuses_count", "user_verified"]

        df_filtered = pd.DataFrame(columns=df_columns)

        files = [f for f in listdir(self.data_path) if isfile(join(self.data_path, f))]

        filters = {"tweet_type": ["original"], "lang": ["fa"]}

        tags = du.get_hashtags()
        tags = [k for k, v in tags.items() if "fa" in v]

        file_name_idx = 0
        for file in files:
            if file.endswith(".xlsx") and not file.startswith("~$"):
                df = pd.read_excel(self.data_path + file, nrows=count)

                for index, row in df.iterrows():
                    if row["tweet_type"] in filters["tweet_type"] and row["lang"] in filters["lang"] and any(
                            tag in row["text"] for tag in tags):
                        df_filtered = df_filtered.append({"id": row["id"], "tweet_url": row["tweet_url"],
                                                          "created_at": row["created_at"],
                                                          "text": clean_persian_tweets(row["text"]),
                                                          "user_description": row["user_description"],
                                                          "user_followers_count": row["user_followers_count"],
                                                          "user_friends_count": row["user_friends_count"],
                                                          "user_location": row["user_location"],
                                                          "user_statuses_count": row["user_statuses_count"],
                                                          "user_verified": row["user_verified"]},
                                                         ignore_index=True)
                        # saving file at checkpoint
                        if len(df_filtered) % save_checkpoint == 0:
                            df_filtered.to_excel(self.cleaned_tweet_path + str(file_name_idx) + ".xlsx",
                                                 engine='xlsxwriter')
                            file_name_idx += 1

                            # reset the data frame
                            df_filtered = pd.DataFrame(columns=df_columns)

        # saving the last file
        if len(df_filtered) > 0:
            df_filtered.to_excel(self.cleaned_tweet_path + str(file_name_idx) + ".xlsx", engine='xlsxwriter')

    def clean_data_json(self, save_checkpoint=100):
        """
        reading tweets' json data
        :param save_checkpoint: saving point of tweets' data frame
        :return:
        """

        df_columns = ["id", "created_at", "text", "user_description", "user_followers_count", "user_friends_count",
                      "user_location", "user_location_carmen", "user_statuses_count", "user_verified"]
        df_filtered = pd.DataFrame(
            columns=df_columns)

        files = [f for f in listdir(self.data_path) if isfile(join(self.data_path, f))]

        filters = {"lang": ["fa"]}

        hashtags = du.get_hashtags()
        tags = []
        for k, v in hashtags.items():
            tags.append(k.replace('\n', '').replace('\r', '').replace('#', '').strip())

        file_name_idx = 0
        for file in files:
            if file.endswith(".json") and not file.startswith("~$"):
                with open(self.data_path + file) as f:
                    for line in f:
                        row = json.loads(line)
                        if "user" in row and row["in_reply_to_status_id_str"] in ["", None] and row["lang"] in filters[
                            "lang"] and 'RT' not in row["text"] and any(tag in row["text"] for tag in tags):

                            # location is created by CARMEN, if resolved
                            if "location" in row and "country" in row["location"] and \
                                    row["location"]["country"] not in [None, "NaN", ""]:
                                user_location = row["location"]["country"]
                            else:
                                user_location = ""
                            df_filtered = df_filtered.append({"id": row["id"],
                                                              "created_at": row["created_at"],
                                                              "text": clean_persian_tweets(row["text"]),
                                                              "user_description": row["user"]["description"],
                                                              "user_followers_count": row["user"]["followers_count"],
                                                              "user_friends_count": row["user"]["friends_count"],
                                                              "user_location": row["user"]["location"],
                                                              "user_location_carmen": user_location,
                                                              "user_statuses_count": row["user"]["statuses_count"],
                                                              "user_verified": row["user"]["verified"]},
                                                             ignore_index=True)
                            # saving file at checkpoint
                            if len(df_filtered) % save_checkpoint == 0:
                                df_filtered.to_excel(self.cleaned_tweet_path + str(file_name_idx) + ".xlsx",
                                                     engine='xlsxwriter')
                                file_name_idx += 1

                                # reset the data frame
                                df_filtered = pd.DataFrame(columns=df_columns)

        # saving the last file
        if len(df_filtered) > 0:
            df_filtered.to_excel(self.cleaned_tweet_path + str(file_name_idx) + ".xlsx", engine='xlsxwriter')


def remove_url(text):
    text = ' '.join(x for x in text.split() if x.startswith('http') == False and x.startswith('www') == False)
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^http?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^www?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    # extra step to make sure html tags are completely removed
    clean = re.compile('<.*>|<.*\"')
    result = re.sub(clean, '', text)
    return result


def emoji_free_text(text):
    return emoji.get_emoji_regexp().sub(r'', text)


def clean_persian_tweets(tweet):
    # removing URLs
    tweet = remove_url(tweet)

    tweet = emoji_free_text(tweet)

    # removing more than one space
    tweet = ' '.join(hazm_docs(tweet).split())

    filtered_tweet = []
    for token in tweet.split():
        if not token.startswith("@") and not token.startswith("https") and not token.startswith(
                "&amp") and token != "RT":
            filtered_tweet.append(token.strip())

    tweet = " ".join(filtered_tweet)

    # removing english characters
    tweet = re.sub(r'[a-zA-Z]', '', tweet)

    # removing non-relevant punctuation marks
    puncs = list("÷!٬٫٪×،*)(ـ+ًٌٍَُِّْ][}{|«»:؛ٰٓ‌ٔ<>؟-=#:|؟!,،?.؛()«»…".replace("‌", ""))
    puncs.extend(string.punctuation)
    puncs = set(puncs)
    for punc in puncs:
        tweet = tweet.replace(punc, " ")

    # removing single characters
    tweet = ' '.join([t for t in tweet.split(" ") if len(t) > 1])

    cleaned_tweet = re.sub(' +', ' ', tweet)
    return cleaned_tweet


def hazm_docs(doc, lemm=False, stem=False):
    """
    processing documents using Persian Hazm library
    :param doc: doc is a string
    :param lemm: True if want to lemmatize, False, otherwise
    :param stem: True if want to stem, False, otherwise
    :return:
    """
    normalized_doc = normalizer.normalize(doc)
    doc_sents = sent_tokenize(normalized_doc)
    words = []
    for sent in doc_sents:
        words.extend(word_tokenize(sent))

    if stem:
        words = [stemmer.stem(t) for t in words]
    if lemm:
        words = [lemmatizer.lemmatize(t) for t in words]

    processed_doc = " ".join(words)

    return processed_doc
