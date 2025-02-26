# **Data Exploration**
The data exploration performed on the cleaned dataset consisted of multiple analyses to uncover various types of information embedded in the data. This exploration is divided intotwo subtasks: general exploration, and text clustering analysis.

## **General Exploration**
The `general_data_exploration_example.ipynb` file contains a general exploratory analysis of the dataset. It includes basic information about data types, along with several visualizations of potentially relevant data features. All functions used in this notebook can be accessed through the `general_data_exploration_visualizations.py` module.

## **Clustering Analysis**
This analysis is documented in the `clustering_analysis_example.ipynb` file. Since the dataset is primarily composed of text, this exploration aims to visualize its data points and provide an unsupervised interpretation using clustering techniques. All functions used in this notebook can be accessed through the `clustering_analysis.py` module.

The notebook offers a step-by-step guide on clustering the abstracts in the dataset. Additionaly, this exploration compares different methods for dimensionality reduction and cluster generation to further analyse the data and the information captured by the clusters. 

Text embeddings are used for this analysis, and GPUs are employed to accelerate computation times.
