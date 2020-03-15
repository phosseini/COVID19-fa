import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from data_reader import DataLoader
from mpl_toolkits.mplot3d import Axes3D


def visualize_cluster(est):
    fig = plt.figure(1, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    labels = est.labels_

    ax.scatter(tfidf[:, 3], tfidf[:, 0], tfidf[:, 2],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title('8 clusters')
    ax.dist = 12

    # Plot the ground truth
    fig = plt.figure(1, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    for name, label in [('Setosa', 0),
                        ('Versicolour', 1),
                        ('Virginica', 2)]:
        ax.text3D(tfidf[clusters == label, 3].mean(),
                  tfidf[clusters == label, 0].mean(),
                  tfidf[clusters == label, 2].mean() + 2, name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(clusters, [1, 2, 0]).astype(np.float)
    ax.scatter(tfidf[:, 3], tfidf[:, 0], tfidf[:, 2], c=y, edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title('Ground Truth')
    ax.dist = 12

    fig.show()


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
save_doc_cluster_file = False
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

    km = MiniBatchKMeans(n_clusters=n_cluster, batch_size=2048, random_state=20)
    kmeans = km.fit(tfidf)
    clusters = kmeans.predict(tfidf)

    get_top_keywords(tfidf, clusters, tfidf_vectorizer.get_feature_names(), 15)

    # visualize_cluster(kmeans)

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
    if save_doc_cluster_file:
        doc_annotation.to_excel("data/doc_annotation.xlsx")
