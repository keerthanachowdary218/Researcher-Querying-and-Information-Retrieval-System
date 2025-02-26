#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.autonotebook import tqdm
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np

def load_cde_model():
    """
    Load CDE tokenizer and model.
    
    Returns:
        cde_tokenizer: Tokenizer for CDE model.
        cde_model: Pre-trained CDE model.
    """
    cde_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", trust_remote_code=True)
    cde_model = AutoModel.from_pretrained("jxm/cde-small-v1", trust_remote_code=True)
    return cde_tokenizer, cde_model

def load_bart_model():
    """
    Load BART finetuned keyphrase extraction tokenizer and model.

    Returns:
        bart_tokenizer: Tokenizer for BART model.
        bart_model: Pre-trained BART model.
    """
    bart_tokenizer = AutoTokenizer.from_pretrained("aglazkova/bart_finetuned_keyphrase_extraction")
    bart_model = AutoModelForSeq2SeqLM.from_pretrained("aglazkova/bart_finetuned_keyphrase_extraction")
    return bart_tokenizer, bart_model

def embed_corpus(corpus, tokenizer, model, document_prefix = "search_document: ", device = torch.device, batch_size = 2):
    """
    Embeds a list of documents using a two-stage embedding model.

    Parameters:
        corpus (list): The list of documents to embed.
        tokenizer: The tokenizer for processing the documents.
        model: The pre-trained model for generating embeddings.
        document_prefix (str): A prefix to add to each document before tokenizing.
        device (torch.device): The device (e.g., CPU or GPU) to run the embeddings.
        batch_size (int): The number of documents to process per batch (default is 2).

    Returns:
        list of tensors: The embeddings of the documents.
    """
    # Tokenize the corpus with the document prefix, truncating and padding to ensure uniform input lengths.
    tokenized_docs = tokenizer(
        [document_prefix + doc for doc in corpus],  # Add the prefix to each document
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    dataset_embeddings = []  # Initialize a list to store embeddings for each batch

    # Process documents in batches to save memory and improve efficiency
    for i in tqdm(range(0, len(tokenized_docs["input_ids"]), batch_size)):
        # Extract a batch of documents from the tokenized inputs
        minicorpus_docs_batch = {k: v[i:i + batch_size] for k, v in tokenized_docs.items()}

        # Compute embeddings using the first stage of the pre-trained model
        with torch.no_grad():  # Disable gradient computation for efficiency
            embeddings = model.first_stage_model(**minicorpus_docs_batch)

        # Append the batch embeddings to the overall list
        dataset_embeddings.append(embeddings)

    # Concatenate all batch embeddings into a single tensor and convert to a list for compatibility
    dataset_embeddings = torch.cat(dataset_embeddings).tolist()

    # Save the embeddings to a DataFrame for easier manipulation and analysis
    embeddings_df = pd.DataFrame({'Embeddings': dataset_embeddings})

    # Serialize the DataFrame to a pickle file for future use
    with open("../../data/dataset_embeddings.pkl", "wb") as file:
        pickle.dump(embeddings_df, file)

    # Return the list of embeddings
    return dataset_embeddings

def batchify(docs, batch_size):
    """
    Splits a list-like data into smaller batches.

    Parameters:
        data (list): The list of items to batch.
        batch_size (int): The number of items in each batch.

    Returns:
        list: A batch of items.
    """
    # Iterate over the input list in steps of `batch_size`
    for i in range(0, len(docs), batch_size):
        # Yield a slice of the list from the current index to the current index + batch_size
        yield docs[i:i + batch_size]

def generate_embeddings_in_batches(column, tokenizer, model, dataset_embeddings, document_prefix = "search_document: "):
    """
    Generate embeddings for a list of documents in batches and return concatenated embeddings.

    Parameters:
        docs (list): List of documents (strings) to embed.
        tokenizer: Tokenizer for the model being used.
        model: Pre-trained model used for generating embeddings.
        document_prefix (str): Prefix to add to each document before tokenizing.
        dataset_embeddings: Pre-computed embeddings from the dataset.
        batch_size (int): Number of documents to process in each batch (default: 2).

    Returns:
        torch.Tensor: Concatenated embeddings for all documents.
    """
    # Convert 'docs' column to a list
    docs = column.to_list()

    # Define batch size
    batch_size = 2

    # Create an empty tensor to store the embeddings for all batches
    all_doc_embeddings = []

    # Iterate through batches
    for batch in batchify(docs, batch_size):
        # Tokenize the current batch of documents
        tokenized_batch = tokenizer(
            [document_prefix + doc for doc in batch],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        ).to(model.device)

        # Generate embeddings for the batch
        with torch.no_grad():
            doc_embeddings = model.second_stage_model(
                input_ids=tokenized_batch["input_ids"],
                attention_mask=tokenized_batch["attention_mask"],
                dataset_embeddings=dataset_embeddings,
            )

        # Normalize embeddings
        doc_embeddings /= doc_embeddings.norm(p=2, dim=1, keepdim=True)

        # Collect the embeddings for the batch
        all_doc_embeddings.append(doc_embeddings)

    # Concatenate the embeddings for all batches
    all_doc_embeddings = torch.cat(all_doc_embeddings, dim=0)

    # Now, `all_doc_embeddings` contains the embeddings for all documents.

    return all_doc_embeddings

def extract_keywords(text, bart_tokenizer, bart_model):
    """
    Extracts keywords from a given text using a sequence-to-sequence model.

    Parameters:
        text (str): The input text from which to extract keywords.
        bart_tokenizer: Tokenizer for BART model.
        bart_model: Pre-trained BART model.

    Returns:
        str: A string containing the extracted keywords.
    """
    # Tokenize the input text and prepare it for the sequence-to-sequence model.
    tokenized_text = bart_tokenizer.prepare_seq2seq_batch([text], return_tensors='pt')
    # Generate the output (extracted keywords) using the sequence-to-sequence model.
    translation = bart_model.generate(**tokenized_text)
    # Since we only have one input, extract the first result from the decoded batch
    translated_text = bart_tokenizer.batch_decode(translation, skip_special_tokens=True)[0]
    return translated_text

def assemble_dataframe(df):
    """
    Processes a DataFrame to generate embeddings and extract keywords for text data.

    Parameters:
        df (pd.DataFrame): A DataFrame containing a column named 'Abstract', 
                           which holds the text data to process.

    Returns:
        pd.DataFrame: The input DataFrame augmented with the following columns:
                      - 'cde-small-v1': CDE-generated embeddings for the abstracts.
                      - 'keywords': Extracted keywords from the abstracts.
                      - 'keyword_embeddings': Embeddings of the extracted keywords.
    """
    bart_tokenizer, bart_model = load_bart_model()
    df['keywords'] = df['Abstract'].apply(extract_keywords, args=(bart_tokenizer, bart_model))
    cde_tokenizer, cde_model = load_cde_model()
    randomSample = df.sample(n = 512, replace = False, random_state = 1) # take a sample of 512 abstracts, of minicorpus_size
    minicorpus_docs = randomSample['Abstract'].to_list()
    dataset_embeddings = embed_corpus(corpus = minicorpus_docs, tokenizer = cde_tokenizer, model = cde_model,
                                      document_prefix = "search_document: ", device = torch.device, batch_size = 2)
    cde_embeddings = generate_embeddings_in_batches(column = df['Abstract'], tokenizer = cde_tokenizer, model = cde_model,
                                                        document_prefix = "search_document: ", dataset_embeddings = dataset_embeddings)
    df['cde-small-v1'] = [embedding.tolist() for embedding in cde_embeddings]
    sbert_model = SentenceTransformer('all-mpnet-base-v2')
    keyword_embeddings = df['keywords'].apply(lambda x: sbert_model.encode(x))
    # Convert embeddings to a DataFrame and add as a column
    df['keyword_embeddings'] = embeddings.tolist()
    return df

def save_to_pickle(df):
    """
    Saves a DataFrame to a pickle file.

    Parameters:
        df (pd.DataFrame): The DataFrame to save.

    Returns:
        None
    """
    with open("../../data/publication_data.pkl", "wb") as file:
        pickle.dump(df, file)

def load_from_pickle(file_name):
    """
    Loads a DataFrame from a pickle file.

    Parameters:
        file_name (str): The name of the pickle file to load.

    Returns:
        pd.DataFrame: The DataFrame loaded from the pickle file.
    """
    with open(file_name, "rb") as file:
        return pickle.load(file)

def cde_embed_query(text: str, tokenizer, model, dataset_embeddings) -> torch.Tensor:
    """
    Embed a query using the CDE model.
    
    Parameters:
        text (str): Input text to embed.
        tokenizer: Tokenizer for CDE model.
        model: Pre-trained CDE model.
        dataset_embeddings: Pre-computed embeddings from the dataset.
    
    Returns:
        torch.Tensor: The vector representation of the input text.
    """
    # Tokenize the query text with truncation and padding, returning PyTorch tensors
    query_prefix = "Query: " # Add a prefix to indicate the text is a query
    queries = [text]
    queries = tokenizer(
        [query_prefix + query for query in queries],
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    ).to(model.device)

    # Compute query embeddings using the model's second stage
    with torch.no_grad():
        query_embeddings = model.second_stage_model(
            input_ids=queries["input_ids"],
            attention_mask=queries["attention_mask"],
            dataset_embeddings=dataset_embeddings,
        )
    # Normalize the embeddings to unit length
    query_embeddings /= query_embeddings.norm(p=2, dim=1, keepdim=True)
    return query_embeddings.flatten() # Flatten the embeddings and return as a single tensor

def cde_retriever(df: pd.DataFrame, input_text: str, tokenizer, model, dataset_embeddings, k = 5) -> pd.DataFrame:
    """
    Retrieve the top k most similar records from the dataframe using CDE embeddings.
    
    Parameters:
        df (pd.DataFrame): DataFrame with CDE embeddings.
        input_text (str): Query text.
        tokenizer: Tokenizer for CDE model.
        model: Pre-trained CDE model.
        dataset_embeddings: Pre-computed dataset embeddings.
        k (int): Number of top records to retrieve.
    
    Returns:
        pd.DataFrame: Top k most similar records based on cosine similarity.
    """
    # Compute the embedding for the input query text
    input_embedding = cde_embed_query(input_text, tokenizer, model, dataset_embeddings)
    # Prepare input embedding as a detached tensor without gradient tracking
    input_embedding_tensor = input_embedding.clone().detach().requires_grad_(False)
    # Stack dataset embeddings from the DataFrame into a tensor
    df_embeddings = torch.stack([torch.tensor(embedding).clone().detach() for embedding in df['cde-small-v1'].values])
    # Normalize the input embedding and the dataset embeddings to unit vectors
    input_embedding_norm = input_embedding_tensor / input_embedding_tensor.norm()
    df_embeddings_norm = df_embeddings / df_embeddings.norm(dim=1, keepdim=True)
    # Compute cosine similarities between the input embedding and dataset embeddings
    similarities = torch.matmul(df_embeddings_norm, input_embedding_norm).numpy()
    # Add the similarity scores to the DataFrame
    df['similarity'] = similarities
    # Retrieve the top k records sorted by similarity score in descending order
    top_k = df.sort_values(by='similarity', ascending=False).head(k)
    return top_k

def sbert_retriever(df, input_embedding, k = 500) -> pd.DataFrame:
    """
    Retrieve the top k most similar records from the dataframe using SentenceBert embeddings.
    
    Parameters:
        df (pd.DataFrame): DataFrame with SentenceBERT embeddings.
        input_embedding (np.array): Query embedding
        k (int): Number of top records to retrieve.
    
    Returns:
        pd.DataFrame: Top k most similar records based on cosine similarity.
    """
    df_embeddings = np.vstack(df['keyword_embeddings'].values)  # Stack embeddings into a 2D array
    similarities = cosine_similarity([input_embedding], df_embeddings)[0] # Compute cosine similarity
    df['similarity'] = similarities
    top_k = df.sort_values(by='similarity', ascending=False).head(k) # sort values by highest cosine similarity
    return top_k

def full_retriever(df, question, dataset_embeddings, sbert_model, cde_tokenizer, cde_model):
    """
    Retrieves relevant entries from a DataFrame based on a given question using a combined approach.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data to search.
        question (str): The query or question for which relevant entries are to be retrieved.
        dataset_embeddings (list): Precomputed embeddings for the dataset to assist with CDE retrieval.
        sbert_model (SentenceTransformer): The SBERT model used to encode the input question for the first retrieval step.
        cde_tokenizer: The tokenizer for the CDE model.
        cde_model: The pre-trained CDE model.

    Returns:
        pd.DataFrame: A subset of the original DataFrame containing entries most relevant to the question.
    """
    return cde_retriever(sbert_retriever(df, sbert_model.encode(question)),
                         input_text = question,
                         tokenizer = cde_tokenizer,
                         model = cde_model,
                         dataset_embeddings = dataset_embeddings)
            
