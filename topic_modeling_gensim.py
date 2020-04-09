import matplotlib.pyplot as plt

from data_reader import DataLoader
from pre_processing import hazm_docs

from gensim import corpora
from gensim.models import LdaModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel

# loading cleaned tweets
params = {"n_count": 1000000}
data = DataLoader().load_tweets(n_count=params["n_count"])
data = data["text"].values.astype('U').tolist()

# removing numbers
for i in range(len(data)):
    data[i] = " ".join(t for t in data[i].split() if not t.strip().isdigit())

# loading persian stop words
with open("data/persian_stop_words.txt", "r") as f:
    persian_stop_words = f.readlines()
persian_stop_words = [x.strip() for x in persian_stop_words]


def learn_lda_model(corpus, dictionary, k):
    lda = LdaMulticore(corpus,
                       workers=3,
                       id2word=dictionary,
                       num_topics=k,
                       random_state=42,
                       iterations=50,
                       per_word_topics=False,
                       eval_every=None)
    cm = CoherenceModel(model=lda, corpus=corpus, coherence='u_mass')
    coherence = cm.get_coherence()
    print('{}: {}'.format(k, coherence))
    return k, coherence, lda


def plot_scores(scores, ax, ylabel):
    _x = [s[0] for s in scores]
    _y = [s[1] for s in scores]

    ax.plot(_x, _y, color='tab:blue')
    ax.set_xlabel('k')
    ax.set_ylabel(ylabel)
    ax.set_title('{} vs k'.format(ylabel))


def remove_stop_words(tokens, stop_words):
    out = [t for t in tokens.split(' ') if t.strip() not in stop_words]
    return out


def find_topics_count():
    lda_scores = [learn_lda_model(corpus, dictionary, k) for k in range(2, 30)]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    plot_scores(lda_scores, ax[2], 'LDA Coherence')

    plt.tight_layout()
    plt.show()


T = [remove_stop_words(t, persian_stop_words) for t in data]
dictionary = corpora.Dictionary(T)
dictionary.filter_extremes(no_below=5, no_above=0.95)
corpus = [dictionary.doc2bow(text) for text in T]
print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))
