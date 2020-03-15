import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from data_reader import DataLoader


def find_optimal_clusters(max_k):
    iters = range(2, max_k + 1, 2)

    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, batch_size=2048, random_state=20).fit(tfidf).inertia_)
        print('Fit {} clusters'.format(k))

    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    plt.show()


def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()

    for i, r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))


n_samples = 1000000
n_features = 400
do_elbow_kmeans = False
do_elbow_mini_kmeans = True
save_doc_cluster_file = True
fit_kmeans = True

# loading data
data = DataLoader().load_data(count=n_samples)

# filtering tweets based on the # of followers
data = data.loc[(data["user_followers_count"] > 10) & (data["user_statuses_count"] > 10)]

data = data["text"].to_list()

data_samples = data[:n_samples] if n_samples >= len(data) else data

print("# of samples: ", len(data_samples))

with open("data/persian_stop_words.txt", "r") as f:
    persian_stop_words = f.readlines()

persian_stop_words = [x.strip() for x in persian_stop_words]

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=5, max_features=n_features, stop_words=persian_stop_words)
tfidf = tfidf_vectorizer.fit_transform(data_samples)

if do_elbow_kmeans:
    Sum_of_squared_distances = []
    K = range(1, 15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(tfidf)
        Sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

if do_elbow_mini_kmeans:
    find_optimal_clusters(20)

if fit_kmeans:
    n_cluster = 8
    n_annotation_samples = 30

    kmeans = MiniBatchKMeans(n_clusters=n_cluster, batch_size=2048, random_state=20).fit(tfidf)
    clusters = MiniBatchKMeans(n_clusters=n_cluster, batch_size=2048, random_state=20).fit_predict(tfidf)

    get_top_keywords(tfidf, clusters, tfidf_vectorizer.get_feature_names(), 15)

    doc_cluster = pd.DataFrame(columns=["doc", "cluster"])
    for doc in data_samples:
        doc_cluster = doc_cluster.append({"doc": doc, "cluster": kmeans.predict(tfidf_vectorizer.transform([doc]))[0]},
                                         ignore_index=True)

    if save_doc_cluster_file:
        doc_cluster.to_excel("data/doc_clusters.xlsx")

    doc_annotation = pd.DataFrame(columns=["doc", "cluster", "label"])

    for i in range(n_cluster):
        samples = doc_cluster.loc[doc_cluster.cluster == i].sample(n_annotation_samples)
        for index, row in samples.iterrows():
            doc_annotation = doc_annotation.append({"doc": row["doc"], "cluster": row["cluster"], "label": ""},
                                                   ignore_index=True)

    doc_annotation.to_excel("data/doc_annotation.xlsx")
