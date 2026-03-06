from sklearn.cluster import KMeans


def cluster_states(features, k=4):
    model = KMeans(n_clusters=k)
    labels = model.fit_predict(features)
    return labels
