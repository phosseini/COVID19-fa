from data_loader import DataLoader

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

n_samples = 1000000
n_features = 1000
n_components = 7
n_top_words = 20


def print_top_words(model, feature_names):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


# loading cleaned tweets
params = {"vectorizer": "tfidf", "n_cluster": 9, "n_count": 1000000}
data = DataLoader().load_tweets(n_count=params["n_count"])
data = data["text"].values.astype('U').tolist()

# removing numbers
for i in range(len(data)):
    data[i] = " ".join(t for t in data[i].split() if not t.strip().isdigit())

# loading persian stop words
with open("../data/persian_stop_words.txt", "r") as f:
    persian_stop_words = f.readlines()
persian_stop_words = [x.strip() for x in persian_stop_words]

# LDA can only use raw term counts because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.9, min_df=5, max_features=n_features, stop_words=persian_stop_words)
tf = tf_vectorizer.fit_transform(data)
tf_feature_names = tf_vectorizer.get_feature_names()

# Run LDA
lda = LatentDirichletAllocation(n_components=n_components, max_iter=5, learning_method='online', learning_offset=50.,
                                random_state=0).fit(tf)

print("LDA Top words:")
print_top_words(lda, tf_feature_names)
