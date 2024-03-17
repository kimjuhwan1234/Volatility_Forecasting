from umap import UMAP
from nltk import WordNetLemmatizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def vectorize_categories(document_df):
    model = SentenceTransformer('all-mpnet-base-v2')
    texts = document_df.tolist()
    embeddings = model.encode(texts, convert_to_numpy=True)
    print(embeddings.shape)

    umap_model = UMAP(n_neighbors=10, min_dist=0.5, n_components=2, random_state=42, metric='cosine')
    reduced_embeddings = umap_model.fit_transform(embeddings)
    return embeddings, reduced_embeddings


def clustering_categories(document_df, embeddings, n_clusters):
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, max_iter=1000).fit(embeddings)
    cluster_label = kmeans.labels_
    document_df = pd.concat([document_df, pd.DataFrame(cluster_label, columns=['cluster'])], axis=1)
    return document_df, cluster_label


def print_reduced_cluster(document_df):
    for i, label in enumerate(list(set(document_df['cluster']))):
        print(label)
        print(set(document_df.iloc[:, 0][document_df['cluster'] == label]))
        num = document_df.iloc[:, 0][document_df['cluster'] == label].count() / len(document_df) * 100
        print(f'{num:.2f}%')
        print()


def plot_UMAP(title, reduced_embeddings, cluster_labels):
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.plasma(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        plt.scatter(reduced_embeddings[cluster_labels == label, 0],
                    reduced_embeddings[cluster_labels == label, 1], color=colors[i], label=label, alpha=0.6)

    plt.title(f"{title}", fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.show()


def LemTokens(tokens):
    lemmar = WordNetLemmatizer()
    return [lemmar.lemmatize(word=token, pos='n') for token in tokens]


def LemNormalize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return LemTokens(nltk.wordpunct_tokenize(text.lower().translate(remove_punct_dict)))


def verterize(df):
    tfidf_vect = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english', ngram_range=(1, 3))
    ftr_vect = tfidf_vect.fit_transform(df)
    print(ftr_vect.shape)
    return ftr_vect
