# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause

from time import time
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from data_reader import DataLoader

n_samples = 1000000
n_features = 1000
n_components = 5
n_top_words = 20


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


# loading data
data = DataLoader().load_data(count=n_samples)

# filtering tweets based on the # of followers
data = data.loc[(data["user_followers_count"] > 10) & (data["user_statuses_count"] > 10)]

data = data["text"].to_list()

if n_samples >= len(data):
    data_samples = data[:n_samples]
else:
    data_samples = data

print("# of samples: ", len(data_samples))

with open("data/persian_stop_words.txt", "r") as f:
    persian_stop_words = f.readlines()

persian_stop_words = [x.strip() for x in persian_stop_words]

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, max_features=n_features, stop_words=persian_stop_words)
tfidf = tfidf_vectorizer.fit_transform(data_samples)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.9, min_df=5, max_features=n_features, stop_words=persian_stop_words)
tf = tf_vectorizer.fit_transform(data_samples)
tf_feature_names = tf_vectorizer.get_feature_names()

# Run NMF
nmf = NMF(n_components=n_components, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# Run LDA
lda = LatentDirichletAllocation(n_components=n_components, max_iter=5, learning_method='online', learning_offset=50.,
                                random_state=0).fit(tf)

print("NFM Top words:")
print_top_words(nmf, tfidf_feature_names, n_top_words)

print("LDA Top words:")
print_top_words(lda, tf_feature_names, n_top_words)
