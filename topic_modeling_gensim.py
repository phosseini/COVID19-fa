import pandas as pd
import matplotlib.pyplot as plt

from data_reader import DataLoader
from pre_processing import hazm_docs

from gensim import corpora
from gensim.models import LdaModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel


class TopicModeling:
    def __init__(self):
        # loading cleaned tweets
        params = {"n_count": 1000}
        data = DataLoader().load_tweets(n_count=params["n_count"])
        data = data["text"].values.astype('U').tolist()

        # removing numbers (not part of pre processing since not always we want to remove numbers)
        for i in range(len(data)):
            data[i] = " ".join(t for t in data[i].split() if not t.strip().isdigit())
        self.data = data

        # loading persian stop words
        with open("data/persian_stop_words.txt", "r") as f:
            persian_stop_words = f.readlines()
        self.persian_stop_words = [x.strip() for x in persian_stop_words]

    def create_corpus(self):
        # TODO: save the corpus and dict and load them if already exist
        T = [self.remove_stop_words(t, self.persian_stop_words) for t in self.data]
        dictionary = corpora.Dictionary(T)
        dictionary.filter_extremes(no_below=5, no_above=0.95)
        corpus = [dictionary.doc2bow(text) for text in T]
        print('Number of unique tokens: %d' % len(dictionary))
        print('Number of documents: %d' % len(corpus))
        return dictionary, corpus

    @staticmethod
    def learn_lda_model(corpus, dictionary, k, cpu_count):
        lda = LdaMulticore(corpus,
                           workers=cpu_count,
                           id2word=dictionary,
                           num_topics=k,
                           random_state=42,
                           iterations=100,
                           per_word_topics=False,
                           eval_every=None)
        cm = CoherenceModel(model=lda, corpus=corpus, coherence='u_mass')
        coherence = cm.get_coherence()
        print('{}: {}'.format(k, coherence))
        return coherence, lda

    @staticmethod
    def plot_scores(scores, ax, ylabel):
        _x = [s[0] for s in scores]
        _y = [s[1] for s in scores]

        ax.plot(_x, _y, color='tab:blue')
        ax.set_xlabel('k')
        ax.set_ylabel(ylabel)
        ax.set_title('{} vs k'.format(ylabel))

    @staticmethod
    def remove_stop_words(tokens, stop_words):
        out = [t for t in tokens.split(' ') if t.strip() not in stop_words]
        return out

    def find_topics_count(self):
        dictionary, corpus = self.create_corpus()
        df = pd.DataFrame(columns=["k", "coherence"])
        for k in range(2, 30):
            coherence, lda = self.learn_lda_model(corpus, dictionary, k, 3)
            df = df.append({"k": k, "coherence": coherence}, ignore_index=True)

        # %% plot k vs. coherence
        df.set_index('k').plot();
        plt.title("LDA topic coherence")
        plt.xlabel("k (number of topics)")
        plt.ylabel("Topic coherence score")
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig('coherence.pdf', format="pdf", pad_inches=2)
        plt.show()
