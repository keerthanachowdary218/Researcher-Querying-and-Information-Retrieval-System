import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------
# Embeddings
# ---------------------------

def load_sbert_model(model_name='all-MiniLM-L6-v2'):
    """
    Load a Sentence-BERT (SBERT) model with GPU support if available.

    Args:
        model_name (str): The name of the SBERT model to load. Default is 'all-MiniLM-L6-v2'.

    Returns:
        model (SentenceTransformer): The loaded SBERT model.
        device (str): The device used for computation ('cuda' or 'cpu').
    """
    # Determine the available device (GPU or CPU).
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load the SBERT model on the determined device.
    model = SentenceTransformer(model_name, device=device)
    return model, device


def compute_sbert_embeddings(df, text_column='Abstract', model_name='all-MiniLM-L6-v2', batch_size=64):
    """
    Compute SBERT embeddings for a specified text column in a DataFrame and store them.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the text data.
        text_column (str): The column name in `df` containing text to encode. Default is 'Abstract'.
        model_name (str): The name of the SBERT model to use. Default is 'all-MiniLM-L6-v2'.
        batch_size (int): The batch size for encoding the text. Default is 64.

    Returns:
        pandas.DataFrame: The input DataFrame with an additional column 'SBERT_embeddings'
        containing the computed embeddings.
    """
    # Load the SBERT model and determine the device.
    model, device = load_sbert_model(model_name)
    # Extract text data, replacing NaN values with empty strings.
    texts = df[text_column].fillna('').tolist()
    # Compute embeddings using the SBERT model.
    embeddings = model.encode(texts, device=device, batch_size=batch_size)
    # Store embeddings in a new column in the DataFrame.
    df['SBERT_embeddings'] = embeddings.tolist()
    return df

# ---------------------------
# 3-Level Clustering (Agglomerative)
# ---------------------------

def three_level_agglomerative_clustering(df, embeddings_column='SBERT_embeddings',
                                         n_clusters1=12, n_clusters2=6, n_clusters3=3,
                                         save_path=None):
    """
    Perform 3-level hierarchical clustering on embeddings and assign cluster labels to the DataFrame,
    using Agglomerative Hierarchical Clustering method.

    Args:
        df (pandas.DataFrame): Input DataFrame containing embeddings for clustering.
        embeddings_column (str): Column name containing the embeddings as a list. Default is 'SBERT_embeddings'.
        n_clusters1 (int): Number of clusters for Level 1 clustering. Default is 12.
        n_clusters2 (int): Number of clusters for Level 2 clustering. Default is 6.
        n_clusters3 (int): Number of clusters for Level 3 clustering. Default is 3.
        save_path (str or None): Path to save the resulting DataFrame as a pickle file. Default is None.

    Returns:
        pandas.DataFrame: The input DataFrame with additional columns ('Level_1', 'Level_2', 'Level_3')
        for hierarchical cluster labels.
    """
    # Convert the embeddings column to a NumPy array.
    data = np.array(df[embeddings_column].tolist())

    # Initialize clustering level columns with None values.
    df['Level_1'] = None
    df['Level_2'] = None
    df['Level_3'] = None

    # Perform Level 1 clustering.
    level_1_clustering = AgglomerativeClustering(n_clusters=n_clusters1)
    df['Level_1'] = level_1_clustering.fit_predict(data)

    # Iterate over each Level 1 cluster to perform further clustering at Levels 2 and 3.
    for i in range(n_clusters1):
        # Mask for rows belonging to the current Level 1 cluster.
        mask_level_1 = df['Level_1'] == i
        subcluster_data = data[mask_level_1]

        # Perform Level 2 clustering if the cluster size is sufficient.
        if len(subcluster_data) >= n_clusters2:
            subcluster_model = AgglomerativeClustering(n_clusters=n_clusters2)
            subcluster_labels = subcluster_model.fit_predict(subcluster_data)
            df.loc[mask_level_1, 'Level_2'] = subcluster_labels

            # Iterate over each Level 2 cluster to perform Level 3 clustering.
            for j in range(n_clusters2):
                # Mask for rows belonging to the current Level 2 cluster.
                mask_level_2 = mask_level_1 & (df['Level_2'] == j)
                subsubcluster_data = data[mask_level_2]

                # Perform Level 3 clustering if the cluster size is sufficient.
                if len(subsubcluster_data) < n_clusters3:
                    continue

                subsubcluster_model = AgglomerativeClustering(n_clusters=n_clusters3)
                subsubcluster_labels = subsubcluster_model.fit_predict(subsubcluster_data)
                subsubcluster_indices = df[mask_level_2].index

                # Assign Level 3 cluster labels to the DataFrame.
                df.loc[subsubcluster_indices, 'Level_3'] = subsubcluster_labels

    # Save the resulting DataFrame to a pickle file if a save path is provided.
    if save_path:
        df.to_pickle(save_path)

    return df

# ---------------------------
# 3-Level Clustering (Spectral)
# ---------------------------

def three_level_spectral_clustering(df, embeddings_column='SBERT_embeddings',
                                    n_clusters1=12, n_clusters2=6, n_clusters3=3,
                                    random_state=42, save_path=None):
    """
    Perform 3-level hierarchical clustering on embeddings and assign cluster labels to the DataFrame,
    using Spectral Clustering method.

    Args:
        df (pandas.DataFrame): Input DataFrame containing embeddings for clustering.
        embeddings_column (str): Column name containing the embeddings as a list. Default is 'SBERT_embeddings'.
        n_clusters1 (int): Number of clusters for Level 1 clustering. Default is 12.
        n_clusters2 (int): Number of clusters for Level 2 clustering. Default is 6.
        n_clusters3 (int): Number of clusters for Level 3 clustering. Default is 3.
        save_path (str or None): Path to save the resulting DataFrame as a pickle file. Default is None.

    Returns:
        pandas.DataFrame: The input DataFrame with additional columns ('Level_1', 'Level_2', 'Level_3')
        for hierarchical cluster labels.
    """
    data = np.array(df[embeddings_column].tolist())

    df['Level_1'] = None
    df['Level_2'] = None
    df['Level_3'] = None

    # Level 1 clustering
    level_1_clustering = SpectralClustering(n_clusters=n_clusters1, affinity='nearest_neighbors', random_state=random_state)
    df['Level_1'] = level_1_clustering.fit_predict(data)

    # For each Level_1 cluster, cluster further into Level_2 and Level_3
    # Perform Level 1 clustering.
    for i in range(n_clusters1):
        mask_level_1 = df['Level_1'] == i
        subcluster_data = data[mask_level_1]

        # Perform Level 2 clustering if the cluster size is sufficient.
        if len(subcluster_data) >= n_clusters2:
            subcluster_model = SpectralClustering(n_clusters=n_clusters2, affinity='nearest_neighbors', random_state=random_state)
            subcluster_labels = subcluster_model.fit_predict(subcluster_data)
            df.loc[mask_level_1, 'Level_2'] = subcluster_labels

            # Perform Level 3 clustering if the cluster size is sufficient.
            for j in range(n_clusters2):
                mask_level_2 = mask_level_1 & (df['Level_2'] == j)
                subsubcluster_data = data[mask_level_2]

                if len(subsubcluster_data) < n_clusters3:
                    continue

                subsubcluster_model = SpectralClustering(n_clusters=n_clusters3, affinity='nearest_neighbors', random_state=random_state)
                subsubcluster_labels = subsubcluster_model.fit_predict(subsubcluster_data)
                subsubcluster_indices = df[mask_level_2].index
                df.loc[subsubcluster_indices, 'Level_3'] = subsubcluster_labels

    # Save the resulting DataFrame to a pickle file if a save path is provided.
    if save_path:
        df.to_pickle(save_path)
    return df

# ---------------------------
# Evaluation Functions
# ---------------------------

def evaluate_spectral_clustering(embeddings, k_range, random_state=42):
    """
    Evaluate spectral clustering over a range of cluster numbers and plot the results.

    Args:
        embeddings (numpy.ndarray): Input data embeddings for clustering.
        k_range (list or range): Range of cluster numbers (k) to evaluate.
        random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
        tuple: Two lists containing the silhouette scores and Calinski-Harabasz scores
               for each k in k_range.
    """
    # Compute a similarity matrix using cosine similarity.
    similarity_matrix = cosine_similarity(embeddings)
    silhouette_scores = []
    calinski_harabasz_scores = []

    # Iterate over the range of cluster numbers.
    for k in k_range:
        # Perform spectral clustering.
        spectral_model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=random_state)
        labels = spectral_model.fit_predict(similarity_matrix)

        # Compute evaluation metrics.
        silhouette_avg = silhouette_score(embeddings, labels)
        ch_score = calinski_harabasz_score(embeddings, labels)

        silhouette_scores.append(silhouette_avg)
        calinski_harabasz_scores.append(ch_score)

        # Print metrics for the current number of clusters.
        print(f'# of Clusters: {k}, Silhouette Score: {silhouette_avg}, CH Score: {ch_score}')

    # Plot evaluation metrics.
    plt.figure(figsize=(12, 6))

    # Plot Silhouette Scores.
    plt.subplot(1, 2, 1)
    plt.plot(k_range, silhouette_scores, marker='o', color='blue')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.grid(True)

    # Plot Calinski-Harabasz Scores.
    plt.subplot(1, 2, 2)
    plt.plot(k_range, calinski_harabasz_scores, marker='o', color='green')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Calinski-Harabasz Score')
    plt.title('Calinski-Harabasz Score vs. Number of Clusters')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return silhouette_scores, calinski_harabasz_scores


def evaluate_agglomerative_clustering(embeddings, k_range):
    """
    Evaluate agglomerative clustering over a range of cluster numbers and plot the results.

    Args:
        embeddings (numpy.ndarray): Input data embeddings for clustering.
        k_range (list or range): Range of cluster numbers (k) to evaluate.

    Returns:
        tuple: Two lists containing the silhouette scores and Calinski-Harabasz scores
               for each k in k_range.
    """
    silhouette_scores = []
    calinski_harabasz_scores = []

    # Iterate over the range of cluster numbers.
    for k in k_range:
        # Perform agglomerative clustering.
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(embeddings)

        # Compute evaluation metrics.
        silhouette_avg = silhouette_score(embeddings, labels)
        ch_score = calinski_harabasz_score(embeddings, labels)

        silhouette_scores.append(silhouette_avg)
        calinski_harabasz_scores.append(ch_score)

        # Print metrics for the current number of clusters.
        print(f'# of Clusters: {k}, Silhouette Score: {silhouette_avg}, CH Score: {ch_score}')

    # Plot evaluation metrics.
    plt.figure(figsize=(12, 6))

    # Plot Silhouette Scores.
    plt.subplot(1, 2, 1)
    plt.plot(k_range, silhouette_scores, marker='o', color='blue')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.grid(True)

    # Plot Calinski-Harabasz Scores.
    plt.subplot(1, 2, 2)
    plt.plot(k_range, calinski_harabasz_scores, marker='o', color='green')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Calinski-Harabasz Score')
    plt.title('Calinski-Harabasz Score vs. Number of Clusters')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return silhouette_scores, calinski_harabasz_scores

# ---------------------------
# Visualization (t-SNE)
# ---------------------------

def add_tsne_coordinates(df, embeddings_column='SBERT_embeddings', random_state=42):
    """
    Compute t-SNE coordinates for dimensionality reduction and add them to the DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame containing embeddings for visualization.
        embeddings_column (str): Column name containing the embeddings as a list. Default is 'SBERT_embeddings'.
        random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
        pandas.DataFrame: The input DataFrame with additional columns ('tsne_x', 'tsne_y')
        containing the 2D t-SNE coordinates.
    """
    # Convert the embeddings column to a NumPy array.
    embeddings = np.array(df[embeddings_column].tolist())
    # Perform t-SNE to reduce embeddings to 2D.
    tsne = TSNE(n_components=2, random_state=random_state)
    embeddings_2d = tsne.fit_transform(embeddings)
    # Add t-SNE coordinates to the DataFrame.
    df['tsne_x'] = embeddings_2d[:, 0]
    df['tsne_y'] = embeddings_2d[:, 1]

    return df


def plot_tsne_clusters(df, x_col='tsne_x', y_col='tsne_y', cluster_col='spectral_clustering_label'):
    """
    Plot a t-SNE visualization of clusters using a scatter plot.

    Args:
        df (pandas.DataFrame): Input DataFrame containing t-SNE coordinates and cluster labels.
        x_col (str): Column name for the x-coordinate of t-SNE. Default is 'tsne_x'.
        y_col (str): Column name for the y-coordinate of t-SNE. Default is 'tsne_y'.
        cluster_col (str): Column name containing cluster labels. Default is 'spectral_clustering_label'.

    Returns:
        None
    """
    # Create a scatter plot of t-SNE coordinates colored by cluster labels.
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=x_col, y=y_col, hue=cluster_col, palette='viridis', data=df, legend='full'
    )
    plt.title('t-SNE Visualization of Clusters')
    plt.xlabel('t-SNE X')
    plt.ylabel('t-SNE Y')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.show()

# ---------------------------
# Keyword Extraction
# ---------------------------

# Custom stop words to exclude from keyword extraction
custom_stop_word_list = [
    "using", "via", "based", "model", "design", "impact", "state", "efficient",
    "method", "study", "©", "2023", "however,", "results", "high", "use",
    "used", "show", "approach", "system", "new", "two", "article", "research"
    ]

def extract_keywords(texts, top_n=5, custom_stop_words=custom_stop_word_list):
    """
    Extract top N keywords from a list of texts using TF-IDF.

    Args:
        texts (list of str): List of text documents to extract keywords from.
        top_n (int): Number of top keywords to extract. Default is 5.
        custom_stop_words (list of str): Additional custom stop words to exclude. Default is `custom_stop_word_list`.

    Returns:
        list of str: A list of top N keywords.
    """
    if not texts:
        return []

    # Combine custom stop words with default English stop words from TfidfVectorizer.
    stop_words = set(custom_stop_words).union(TfidfVectorizer(stop_words='english').get_stop_words())

    # Create a TF-IDF vectorizer with a maximum of 1000 features.
    vectorizer = TfidfVectorizer(max_features=1000, stop_words=list(stop_words))

    # Fit and transform the texts to compute the TF-IDF matrix.
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Get feature names and their corresponding scores.
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1

    # Sort the keywords by their scores and return the top N.
    keyword_indices = scores.argsort()[-top_n:][::-1]
    return [feature_names[i] for i in keyword_indices]


def assign_keywords_to_clusters(df, text_column='Abstract',
                                level_1_col='Level_1', level_2_col='Level_2', level_3_col='Level_3',
                                top_n=10, save_path=None):
    """
    Assign top keywords to each cluster level in the DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame containing clusters and text data.
        text_column (str): Column name containing the text data. Default is 'Abstract'.
        level_1_col (str): Column name for Level-1 clusters. Default is 'Level_1'.
        level_2_col (str): Column name for Level-2 clusters. Default is 'Level_2'.
        level_3_col (str): Column name for Level-3 clusters. Default is 'Level_3'.
        top_n (int): Number of top keywords to extract for each cluster. Default is 10.
        save_path (str or None): Path to save the resulting DataFrame as a pickle file. Default is None.

    Returns:
        pandas.DataFrame: The input DataFrame with new columns for cluster keywords at each level.
    """
    # Initialize new columns for keywords at each cluster level.
    df['Level_1_keywords'] = None
    df['Level_2_keywords'] = None
    df['Level_3_keywords'] = None

    # Assign keywords to each Level-1 cluster.
    for cluster in df[level_1_col].dropna().unique():
        cluster_texts = df[df[level_1_col] == cluster][text_column].tolist()
        level_1_keywords = extract_keywords(cluster_texts, top_n=top_n)
        df.loc[df[level_1_col] == cluster, 'Level_1_keywords'] = ', '.join(level_1_keywords)

        # Assign keywords to each Level-2 cluster within the Level-1 cluster.
        for subcluster in df[df[level_1_col] == cluster][level_2_col].dropna().unique():
            subcluster_texts = df[(df[level_1_col] == cluster) & (df[level_2_col] == subcluster)][text_column].tolist()
            level_2_keywords = extract_keywords(subcluster_texts, top_n=top_n)
            df.loc[(df[level_1_col] == cluster) & (df[level_2_col] == subcluster), 'Level_2_keywords'] = ', '.join(level_2_keywords)

            # Assign keywords to each Level-3 cluster within the Level-2 cluster.
            for subsubcluster in df[(df[level_1_col] == cluster) & (df[level_2_col] == subcluster)][level_3_col].dropna().unique():
                subsubcluster_texts = df[(df[level_1_col] == cluster) & (df[level_2_col] == subcluster) & (df[level_3_col] == subsubcluster)][text_column].tolist()
                level_3_keywords = extract_keywords(subsubcluster_texts, top_n=top_n)
                df.loc[(df[level_1_col] == cluster) & (df[level_2_col] == subcluster) & (df[level_3_col] == subsubcluster), 'Level_3_keywords'] = ', '.join(level_3_keywords)

    # Save the DataFrame if a save path is provided.
    if save_path:
        df.to_pickle(save_path)

    return df


def print_top_keywords_per_level1(df, level_1_col='Level_1', level_1_keywords_col='Level_1_keywords'):
    """
    Print the top keywords for each Level-1 cluster.

    Args:
        df (pandas.DataFrame): Input DataFrame containing cluster keywords.
        level_1_col (str): Column name for Level-1 clusters. Default is 'Level_1'.
        level_1_keywords_col (str): Column name containing keywords for Level-1 clusters. Default is 'Level_1_keywords'.

    Returns:
        None
    """
    unique_clusters = df[level_1_col].dropna().unique()
    print("\nTop 10 Keywords per Level-1 Cluster:\n")
    for cluster in unique_clusters:
        keywords = df[df[level_1_col] == cluster][level_1_keywords_col].iloc[0]
        print(f"Cluster {cluster}: {keywords}")

# ---------------------------
# Hierarchical Visualization with Plotly
# ---------------------------

def visualize_three_level_clusters(df,
                                   level_1_col='Level_1',
                                   level_2_col='Level_2',
                                   level_3_col='Level_3',
                                   prof_col='Professor',
                                   save_path=None):
    """
    Visualize the 3-level clustering structure using a network graph for each Level-1 cluster.

    Args:
        df (pandas.DataFrame): DataFrame containing clustering levels and metadata.
        level_1_col (str): Column name for Level-1 cluster labels. Default is 'Level_1'.
        level_2_col (str): Column name for Level-2 cluster labels. Default is 'Level_2'.
        level_3_col (str): Column name for Level-3 cluster labels. Default is 'Level_3'.
        prof_col (str): Column name for professor-related data. Default is 'Professor'.
        save_path (str or None): Path to save the visualization as an HTML file. Default is None.

    Returns:
        plotly.graph_objects.Figure: The generated figure with subplots for each Level-1 cluster.
    """
    # Get the unique Level-1 clusters and compute layout dimensions.
    level_1_clusters = df[level_1_col].unique()
    num_clusters = len(level_1_clusters)
    cols = 3
    rows = (num_clusters // cols) + (num_clusters % cols > 0)

    # Create a subplot grid for the visualization.
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"Cluster {cluster}" for cluster in level_1_clusters],
        horizontal_spacing=0.1, vertical_spacing=0.1
    )

    # Iterate through each Level-1 cluster.
    for idx, level_1_cluster in enumerate(level_1_clusters):
        G_sub = nx.Graph()
        level_1_node = f"Cluster {level_1_cluster}"
        level_1_mask = df[level_1_col] == level_1_cluster

        # Gather metadata for the Level-1 node.
        keywords_level_1 = df[level_1_mask]['Level_1_keywords'].iloc[0]
        professor_dict = df[level_1_mask][prof_col].value_counts().to_dict()
        top_professors = sorted(professor_dict.items(), key=lambda x: x[1], reverse=True)[:5]

        # Add the Level-1 node to the graph.
        G_sub.add_node(level_1_node, size=55,
                       color='lightblue',
                       hover_text=f"{level_1_node}<br>Keywords: {keywords_level_1}<br>Professors: {top_professors}")

        # Iterate through Level-2 clusters within the current Level-1 cluster.
        for level_2_cluster in df[level_1_mask][level_2_col].dropna().unique():
            level_2_mask = level_1_mask & (df[level_2_col] == level_2_cluster)
            level_2_node = f"Cluster {level_1_cluster}.{level_2_cluster}"
            keywords_level_2 = df[level_2_mask]['Level_2_keywords'].iloc[0]
            professor_dict = df[level_2_mask][prof_col].value_counts().to_dict()
            top_professors = sorted(professor_dict.items(), key=lambda x: x[1], reverse=True)[:5]

            # Gather metadata for the Level-2 node.
            keywords_level_2 = df[level_2_mask]['Level_2_keywords'].iloc[0]
            professor_dict = df[level_2_mask][prof_col].value_counts().to_dict()
            top_professors = sorted(professor_dict.items(), key=lambda x: x[1], reverse=True)[:5]

            # Add the Level-2 node and edge to the graph.
            G_sub.add_node(level_2_node, size=35, color='orange',
                           hover_text=f"{level_2_node}<br>Keywords: {keywords_level_2}<br>Professors: {top_professors}")
            G_sub.add_edge(level_1_node, level_2_node)

            # Iterate through Level-3 clusters within the current Level-2 cluster.
            for level_3_cluster in df[level_2_mask][level_3_col].dropna().unique():
                level_3_mask = level_2_mask & (df[level_3_col] == level_3_cluster)
                level_3_node = f"Cluster {level_1_cluster}.{level_2_cluster}.{level_3_cluster}"

                # Gather metadata for the Level-3 node.
                keywords_level_3 = df[level_3_mask]['Level_3_keywords'].iloc[0]
                professor_dict = df[level_3_mask][prof_col].value_counts().to_dict()
                top_professors = sorted(professor_dict.items(), key=lambda x: x[1], reverse=True)[:5]

                # Add the Level-3 node and edge to the graph.
                G_sub.add_node(level_3_node, size=25, color='green',
                               hover_text=f"{level_3_node}<br>Keywords: {keywords_level_3}<br>Professors: {top_professors}")
                G_sub.add_edge(level_2_node, level_3_node)

        # Compute positions for nodes in the graph using spring layout.
        pos = nx.spring_layout(G_sub, seed=13, k=0.2)

        # Prepare node and edge data for Plotly visualization.
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        for node, data in G_sub.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(data['hover_text'])
            node_color.append(data['color'])
            node_size.append(data['size'])

        edge_x, edge_y = [], []
        for edge in G_sub.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Create traces for edges and nodes.
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            text=node_text,
            hoverinfo='text',
            marker=dict(
                showscale=False,
                color=node_color,
                size=node_size,
                line_width=2
            )
        )

        # Determine subplot position and add traces.
        row, col = divmod(idx, cols)
        fig.add_trace(edge_trace, row=row + 1, col=col + 1)
        fig.add_trace(node_trace, row=row + 1, col=col + 1)

    # Update figure layout settings.
    fig.update_layout(
        title='3-Level Hierarchical Clustering Visualization',
        title_x=0.5,
        height=rows * 600,
        width=cols * 600,
        showlegend=False
    )

    # Save the figure as an HTML file if a save path is provided.
    if save_path:
        fig.write_html(save_path)

    return fig

# ---------------------------
# Dinemsionality Reduction
# ---------------------------
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

def plot_dimensionality_reduction(X_tsne, df):
    """
    Visualizes PCA, UMAP, and t-SNE results side by side.

    Parameters:
        X_tsne (np.array): t-SNE reduced data.

    Returns:
        None: Displays the side-by-side plots.
    """
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue='Level_1', palette='viridis', data=df, legend='full')
    plt.title('t-SNE Visualization (SBERT embeddings) of Spectral Clusters')
    plt.xlabel('t-SNE X')
    plt.ylabel('t-SNE Y')
    plt.legend(title='Cluster')
    plt.show()
