"""Funciones relacionadas con la reduccion dimensional de los datos"""
import numpy as np, seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

"""Hace analisis tsne"""
def draw_tsne(data, classification, dirOutExp, mode='cluster'):
    np.random.seed(0)
    tsne_results = TSNE(n_components=2, perplexity=40, n_iter=300).fit_transform(data)
    plt.figure(figsize=(16, 10))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1],hue=classification, palette="deep")
    if mode == 'cluster': plt.savefig(dirOutExp + '/' + str(classification.max() + 1) + '_clusters_tsne_classification.png')
    else: plt.savefig(dirOutExp + '/'+mode+'_tsne_classification.png')
    plt.close()

"""hace el analisis pca"""
def draw_pca(data, classification, dirOutExp, mode='cluster'):
    np.random.seed(0)
    tsne_results = PCA(n_components=2).fit_transform(data)
    plt.figure(figsize=(16, 10))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1],hue=classification, s=10, palette="deep")
    if mode == 'cluster': plt.savefig(dirOutExp+'/'+str(classification.max()+1)+'_clusters_pca_classification.png')
    else: plt.savefig(dirOutExp + '/'+mode+'_pca_classification.png')
    plt.close()

"""Dibuja un dendograma de los datos"""
def draw_dendogram(data, dirOutExp):
    plt.figure(figsize=(16, 10))
    linkage_data = linkage(data, method='ward', metric='euclidean')
    dendrogram(linkage_data, truncate_mode='level', p=4)
    plt.savefig(dirOutExp + '/dendogram_classification.png')
    plt.close()