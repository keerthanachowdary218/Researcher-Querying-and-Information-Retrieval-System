"""
In this module we explore some techniques for clustering papers in our
data set based on their titles. First we test some vectorization (embedding)
techniques, and then we move forward with dimensionality reduction and
clustering of the data.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sentence_transformers import SentenceTransformer
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import torch


def generate_embeddings(df, text_column='Abstract', model_name='all-MiniLM-L6-v2', batch_size=64):
    """
    Generates embeddings for a specified text column using a Sentence-BERT model.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        text_column (str): The column containing the text to embed (default is 'Abstract').
        model_name (str): The name of the Sentence-BERT model to use (default is 'all-MiniLM-L6-v2').
        batch_size (int): The batch size for the encoding process (default is 64).

    Returns:
        np.array: An array of embeddings corresponding to the text data.
    """
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the Sentence-BERT model
    model = SentenceTransformer(model_name, device=device)

    # Perform encoding, filling NaN values with empty strings
    embeddings = model.encode(df[text_column].fillna('').tolist(), device=device, batch_size=batch_size)

    return embeddings


def fit_pca(embeddings, n_components=2):
    """
    Applies PCA to reduce dimensionality.

    Parameters:
        embeddings (np.array): The embeddings to reduce in dimensionality.
        n_components (int): Number of components to keep (default is 2).

    Returns:
        np.array: Reduced dimensionality data from PCA.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(embeddings)
    return X_pca


def fit_umap(embeddings, n_components=2, random_state=42):
    """
    Applies UMAP to reduce dimensionality.

    Parameters:
        embeddings (np.array): The embeddings to reduce in dimensionality.
        n_components (int): Number of components to keep (default is 2).
        random_state (int): Random state for reproducibility (default is 42).

    Returns:
        np.array: Reduced dimensionality data from UMAP.
    """
    umap_reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    X_umap = umap_reducer.fit_transform(embeddings)
    return X_umap


def fit_tsne(embeddings, n_components=2, random_state=42, perplexity=30):
    """
    Applies t-SNE to reduce dimensionality.

    Parameters:
        embeddings (np.array): The embeddings to reduce in dimensionality.
        n_components (int): Number of components to keep (default is 2).
        random_state (int): Random state for reproducibility (default is 42).
        perplexity (int): Perplexity parameter for t-SNE (default is 30).

    Returns:
        np.array: Reduced dimensionality data from t-SNE.
    """
    tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)
    X_tsne = tsne.fit_transform(embeddings)
    return X_tsne


def plot_dimensionality_reduction(X_pca, X_umap, X_tsne):
    """
    Visualizes PCA, UMAP, and t-SNE results side by side.

    Parameters:
        X_pca (np.array): PCA reduced data.
        X_umap (np.array): UMAP reduced data.
        X_tsne (np.array): t-SNE reduced data.

    Returns:
        None: Displays the side-by-side plots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # PCA plot
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c='blue', alpha=0.7)
    axes[0].set_title('PCA')

    # UMAP plot
    axes[1].scatter(X_umap[:, 0], X_umap[:, 1], c='green', alpha=0.7)
    axes[1].set_title('UMAP')

    # t-SNE plot
    axes[2].scatter(X_tsne[:, 0], X_tsne[:, 1], c='red', alpha=0.7)
    axes[2].set_title('t-SNE')

    # Customize plots
    for ax in axes:
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def perform_kmeans_clustering(X_umap, n_clusters=30, random_state=42):
    """
    Applies KMeans clustering to the UMAP-reduced data and visualizes the results.

    Parameters:
        X_umap (np.array): UMAP-reduced data.
        n_clusters (int): The number of clusters for KMeans (default is 30).
        random_state (int): Random state for reproducibility (default is 42).

    Returns:
        tuple: Contains cluster labels and silhouette score.
    """
    # Step 3: Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans_labels = kmeans.fit_predict(X_umap)

    # Step 4: Evaluate clustering quality
    silhouette_avg = silhouette_score(X_umap, kmeans_labels)
    print(f'Silhouette Score: {silhouette_avg:.4f}')

    # Step 5: Visualize clusters and centroids
    plt.figure(figsize=(10, 8))
    # Scatter plot of the data points colored by cluster label
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)

    # Plot centroids
    centroids = kmeans.cluster_centers_  # Get the centroids from KMeans
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50, marker='o', label='Cluster Centroids')

    # Plot customization
    plt.title('KMeans Clustering with UMAP and Centroids')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.grid(True)
    plt.show()

    return kmeans_labels, silhouette_avg


def plot_hierarchical_clustering_dendrogram(embeddings, method='ward', truncate_level=5):
    """
    Creates a dendrogram for hierarchical clustering.

    Parameters:
        embeddings (np.array): The embeddings to perform hierarchical clustering on.
        method (str): The linkage method to use (default is 'ward').
        truncate_level (int): The level to truncate the dendrogram (default is 5).

    Returns:
        None: Displays the dendrogram plot.
    """
    # Step 3: Create linkage matrix for the dendrogram using SciPy
    linkage_matrix = linkage(embeddings, method=method)

    # Step 4: Plot the dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix, truncate_mode='level', p=truncate_level)  # Adjust 'p' to limit depth of the tree
    plt.title('Hierarchical Clustering Dendrogram (Truncated)')
    plt.xlabel('Sample index or (cluster size)')
    plt.ylabel('Distance')
    plt.show()


def perform_agglomerative_clustering(embeddings, n_clusters=30):
    """
    Performs Agglomerative Clustering on the given embeddings and assigns cluster labels.

    Parameters:
        embeddings (np.array): The embeddings to cluster.
        n_clusters (int): The number of clusters for Agglomerative Clustering (default is 30).
        df (pd.DataFrame, optional): DataFrame to store cluster labels (default is None).
        cluster_column (str): The name of the column to store cluster labels in the DataFrame (default is 'Cluster_Agglomerative').

    Returns:
        tuple: Contains the cluster labels and the DataFrame (if provided).
    """
    # Step: Perform Agglomerative Clustering
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agglomerative.fit_predict(embeddings)

    return labels
