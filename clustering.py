import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from data_loader import DataLoader

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer


class Clustering:
    def __init__(self):
        # vectorizer can be either "tf" or "tfidf"
        self.params = {"vectorizer": "tfidf",
                       "n_cluster": 9,
                       "n_count": 1000000,
                       "n_annotation_samples": 30,
                       "k_range": (2, 20),
                       "find_optimal_k": False,
                       "save_doc_cluster_file": False,
                       "fit_kmeans": True,
                       "plot_clusters": False}

        data = DataLoader().load_tweets(n_count=self.params["n_count"])
        data = data["text"].values.astype('U').tolist()

        # removing numbers (not part of pre processing since not always we want to remove numbers)
        for i in range(len(data)):
            data[i] = " ".join(t for t in data[i].split() if not t.strip().isdigit())
        self.data = data

        # loading persian stop words
        with open("data/persian_stop_words.txt", "r") as f:
            persian_stop_words = f.readlines()
        self.persian_stop_words = [x.strip() for x in persian_stop_words]

    @staticmethod
    def visualize_clusters(tf_idf_matrix, labels):
        labels_color_map = {
            0: '#20b2aa', 1: '#ff7373', 2: '#ffe4e1', 3: '#005073', 4: '#4d0404',
            5: '#ccc0ba', 6: '#4700f9', 7: '#f6f900', 8: '#00f91d', 9: '#da8c49'
        }
        pca_num_components = 2
        tsne_num_components = 2

        X = tf_idf_matrix.todense()

        reduced_data = PCA(n_components=pca_num_components).fit_transform(X)

        fig, ax = plt.subplots()
        for index, instance in enumerate(reduced_data):
            # print instance, index, labels[index]
            pca_comp_1, pca_comp_2 = reduced_data[index]
            color = labels_color_map[labels[index]]
            ax.scatter(pca_comp_1, pca_comp_2, c=color)
        plt.show()

        # t-SNE plot
        embeddings = TSNE(n_components=tsne_num_components)
        Y = embeddings.fit_transform(X)
        plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
        plt.show()

    @staticmethod
    def find_optimal_k(max_k, fitted_vectorizer, model='minikmeans'):
        """
        finding the optimal number of clusters
        :param max_k: maximum number of clusters to try
        :param model: could either "minikmeans" or "kmeans"
        :return:
        """
        iters = range(2, max_k + 1, 2)

        sse = []
        for k in iters:
            if model == "minikmeans":
                sse.append(
                    MiniBatchKMeans(n_clusters=k, batch_size=2048, random_state=42).fit(fitted_vectorizer).inertia_)
            elif model == "kmeans":
                sse.append(KMeans(n_clusters=k, random_state=42).fit(fitted_vectorizer).inertia_)
            else:
                raise Exception
            print('Fit {} clusters'.format(k))

        f, ax = plt.subplots(1, 1)
        ax.plot(iters, sse, marker='o')
        ax.set_xlabel('Cluster Centers')
        ax.set_xticks(iters)
        ax.set_xticklabels(iters)
        ax.set_ylabel('SSE')
        ax.set_title('SSE by Cluster Center Plot')
        plt.show()

    def find_optimal_kelbow(self, fitted_vectorizer):
        # Instantiate the clustering model and visualizer
        model = MiniBatchKMeans(random_state=42)
        visualizer = KElbowVisualizer(model, k=self.params["k_range"])

        visualizer.fit(fitted_vectorizer)  # Fit the data to the visualizer
        visualizer.show(outpath="kelbow_minibatchkmeans.pdf")  # Finalize and render the figure

    def fit_vectorizer(self):
        # fitting the vectorizer
        if self.params["vectorizer"] == "tfidf":
            vectorizer = TfidfVectorizer(max_df=0.95, min_df=5, stop_words=self.persian_stop_words)
        elif self.params["vectorizer"] == "tf":
            vectorizer = CountVectorizer(max_df=0.95, min_df=5, stop_words=self.persian_stop_words)
        else:
            raise Exception

        fitted_vectorizer = vectorizer.fit_transform(self.data)

        return fitted_vectorizer, vectorizer

    def fit_kmeans(self, fitted_vectorizer, vectorizer):
        print("Fitting clustering algorithm using: {}".format(self.params["vectorizer"]))
        km = MiniBatchKMeans(n_clusters=self.params["n_cluster"], random_state=42)
        y = km.fit(fitted_vectorizer)

        print("Top terms per cluster:")
        n_words = 15
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        for i in range(self.params["n_cluster"]):
            print("Cluster %d:" % i, end='')
            terms_list = []
            for ind in order_centroids[i, :n_words]:
                terms_list.append(terms[ind])
            print(terms_list)

        # visualizing the tf-idf vectors
        # if plot_clusters:
        #    visualize_clusters(fitted_vectorizer, clusters)

        # doc_cluster = pd.DataFrame(columns=["doc", "cluster"])
        # for doc in data:
        #    doc_cluster = doc_cluster.append({"doc": doc, "cluster": kmeans.predict(vectorizer.transform([doc]))[0]},
        #                                     ignore_index=True)

        # if save_doc_cluster_file:
        #    doc_cluster.to_excel("data/annotation/doc_clusters.xlsx")

        # doc_annotation = pd.DataFrame(columns=["doc", "cluster", "label"])

        # for i in range(params["n_cluster"]):
        #    samples = doc_cluster.loc[doc_cluster.cluster == i].sample(n_annotation_samples)
        #    for index, row in samples.iterrows():
        #        doc_annotation = doc_annotation.append({"doc": row["doc"], "cluster": row["cluster"], "label": ""},
        #                                               ignore_index=True)
        # if save_doc_cluster_file:
        #    doc_annotation.to_excel("data/annotation/doc_annotation.xlsx")
