# Information Retrieval

This repository contains a Jupyter Notebook that utilizes our information retrieval (IR) system. (information_retrieval_demo.ipynb). The example focuses on one sample question we used in our testing, but can be edited to any new question of interest. All the functions used in the example are defined in the module file (information_retrieval.py).

## File Structure

- **information_retrieval.py**: Python file that provides functions necessary for notebook
- **information_retrieval_demo.ipynb**: The main notebook that includes a building of the database and a query-return example

## Key Features

### Information Retrieval System
The IR system akes a user query and returns the 5 most relevant publications to that query. The process is described below:

1. A user query, denoted as Q, is received.

2. The query Q is embedded using the Sentence-BERT model.

3. Cosine similarity is calculated between the embedding of Q and each abstract in D.

4. The top 500 publications are identified based on the highest cosine similarity between the keyphrases of their abstracts and the Sentence-BERT embedding of Q. These publications are collectively referred to as J.

5. The query Q is then embedded using the cde-small-v1 model.

6. The top 5 publications are identified based on the highest cosine similarity between their abstracts and the cde-small-v1 embedding of Q.

Keywords and embeddings for each abstract in the dataset are computed before usage of the system. This includes the BART model that extracts keywords from each abstract, Sentence-BERT which embeds each abstract's keywords, and cde-small-v1 which embeds each abstract directly.

## Usage

 Run the notebook cells to:
   - Assmble the dataframe with keywords and embeddings (Warning: process may take 10+ hours; not necessary to use information retrieval system)
   - Use the information retrieval system to get relevant publications to a query

## Outputs

- **Information Retrieval System**:
  - Relevant publications (includes professor name, abstract, publication year, etc.)
