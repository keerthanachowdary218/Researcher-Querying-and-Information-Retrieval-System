# Researcher Querying and Information Retrieval System

This project aims to build a Researcher Querying and Information Retrieval System for the Office of Corporate and Foundation Relations (OCFR) at Rice University. The system leverages natural language processing (NLP) techniques to connect companies and external stakeholders with the most appropriate faculty members based on their research publications. Our goal is to streamline the process of identifying and facilitating research collaborations by employing word embeddings, knowledge graphs, and machine learning models.

## Project Description

The Researcher Querying and Information Retrieval System is designed to assist the OCFR by querying a database of faculty publications to match external interests with relevant research expertise. This system combines natural language processing, embeddings, and visualization techniques such as knowledge graphs to offer an efficient solution for matching research interests.

Key features include:

- Natural language queries to retrieve faculty profiles and research publications.
- An intuitive visualization tool using knowledge graphs to explore faculty research trends and relationships.
- Integration of SentenceBERT and word embeddings for enhanced understanding of textual data

System Components
1. Clustering Area
Groups research publications and faculty profiles based on similar topics or research domains.
Utilizes embeddings (e.g., BERT, SciBERT) to identify meaningful patterns in the textual data.
Helps visualize clusters within the knowledge graph, making it easier to explore research trends.
![image](https://github.com/user-attachments/assets/3a37f8af-55fb-4c35-bb80-606142da9019)


2. Information Retrieval (IR) Area
Processes natural language queries to match external interests with relevant faculty profiles and publications.
Employs IR techniques to efficiently search and rank results based on the query.
Integrates with the clustering area to refine and visualize search results in context.
![image](https://github.com/user-attachments/assets/d433bc8f-153e-47eb-916a-bfa162dbc484)

This combination of clustering and IR enhances the system's ability to accurately and intuitively connect research interests with relevant expertise.

![image](https://github.com/user-attachments/assets/afaca84d-fef7-493c-9094-307426262077)



## Software Dependencies and Installation
To run this project, you'll need the following dependencies:

### Dependencies:
1. Required python --version: Python 3.11.10
2. Package requirements: requirements.txt

## Setting Up The Environment

**Step 1:** Navigate to your local directory 

**Step 2:** Clone the repository 
    Use the command `git clone https://github.com/RiceD2KLab/OCFR_F24.git`
    
**Step 3:** Check your python version
    1. Use the command `python3 --version`
    2. If the version is `Python 3.11.10`
    3. Otherwise install `python 3.11.10` using the instructions at this [link](https://pythontest.com/python/installing-python-3-11/).
    
**Step 3:** Install the required packages
    1. Update pip: `$ pip install pip update`
    2. With the virtual environment active: `$ pip install -r requirements.txt`


## Data

### Populating Data

**Step 1:** The data required for this project can be downloaded from [here](https://drive.google.com/drive/u/0/folders/1wTMsOeB5t26_yR9pa3YUOzcoBDNncSYV). 

**Step 2:** Place the files in this directory into the `data` folder in this directory. 

### Dataset Overview:
This project uses data sourced from Google Scholar, including publications authored by professors from Rice University's Engineering Department. 

- Number of Papers: 19,919
- Number of Professors: 227
- Average Papers per Professor: 88

## Repository Structure

This repository follows a typical data science workflow consisting of four key stages: Data Scraping, Data Cleaning, Data Exploration, and Modeling. The folders are organized to reflect each stage of the pipeline, ensuring a logical flow and ease of navigation.

![image](https://github.com/user-attachments/assets/3e0a72d8-ec19-4f5c-ad47-9eaafc7dbdee)

1. Data Scraping

Collect raw publication data from Google Scholar using various APIs. This step involves extracting relevant information that will serve as the foundation for analysis.

2. Data Cleaning

Process and clean the raw data by handling missing values, removing duplicates and non-english values. The goal is to prepare the data for accurate analysis.

3. Data Exploration

Experiment with different embedding models such as BERT and SciBERT to represent textual data effectively. Additionally, apply various clustering techniques like hierarchical clustering to identify patterns and group related research publications. The insights gained here inform the subsequent modeling phase.

4. Modeling

Perform clustering to visualize research areas and relationships among publications. Develop an information retrieval (IR) system to extract relevant faculty profiles and publications based on user queries. This step integrates clustering results and embeddings to enhance search accuracy and visualization.



