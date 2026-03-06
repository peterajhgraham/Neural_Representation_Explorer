from sklearn.decomposition import PCA
import umap


def compute_pca(features, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(features)


def compute_umap(features):
    reducer = umap.UMAP()
    return reducer.fit_transform(features)
