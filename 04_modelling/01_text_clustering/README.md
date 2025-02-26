# Text Clustering: Hierarchical Clustering Visualization and Analysis

This repository contains a Jupyter Notebook that demonstrates hierarchical clustering and visualization techniques using Python (`hierarchical_clustering_example.ipynb`). The example focuses on multi-level clustering and visualizing the relationships between clusters using Plotly and NetworkX. All the functions used in the example are defined in the module file (`clustering_module.py`).

## File Structure

- **hierarchical_clustering_example.ipynb**: The main notebook that includes examples of:
  - Keyword extraction for clusters.
  - Multi-level clustering (3-level hierarchy).
  - Visualization of hierarchical clustering using Plotly and NetworkX.

## Key Features

### 1. Multi-Level Clustering
- **3-Level Hierarchical Clustering**:
  - Agglomerative clustering and Spectral clustering are used to identify hierarchical relationships between data points.

### 2. Keyword Extraction
- Uses TF-IDF to extract the most relevant keywords for each cluster level.
- Customizable stop word list to improve the quality of extracted keywords.

### 3. Visualization
- **Network Graph Visualization**:
  - Displays the hierarchical structure of clusters.
  - Interactive plots using Plotly.
- **t-SNE Embedding Visualization**:
  - Projects high-dimensional data into two dimensions for cluster analysis.


## Usage

 Run the notebook cells to:
   - Perform clustering on sample data.
   - Extract keywords for clusters.
   - Visualize the hierarchical clustering structure.

## Outputs

- **Hierarchical Visualization**:
  - Interactive network graph for exploring cluster relationships.
- **Cluster Keywords**:
  - Top keywords for each cluster level (Level 1, Level 2, Level 3).
- **t-SNE Visualization**:
  - 2D representation of high-dimensional data for cluster exploration.

