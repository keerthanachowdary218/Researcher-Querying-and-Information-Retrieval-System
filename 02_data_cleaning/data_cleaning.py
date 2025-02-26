# -*- coding: utf-8 -*-
"""
# Data Cleaning
This script contains the steps for cleaning a dataset, including dropping irrelevant columns, filtering for English text, handling missing values, and retaining only the most recent abstracts for each professor.
"""

import pandas as pd
from langdetect import detect, LangDetectException


def keep_only_english(df):
    """
    Filters the DataFrame to retain only rows where the 'Abstract' column is in English.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing an 'Abstract' column with text data in various languages.

    Returns:
        pandas.DataFrame: A filtered DataFrame containing only rows where the 'Abstract' is in English.
                          Rows where the language is not detected as English will be removed.
    """
    def is_english(text):
        try:
            return detect(text) == 'en'
        except LangDetectException:
            return False

    # Apply language detection to the 'Abstract' column
    df['is_english'] = df['Abstract'].apply(is_english)
    df_english = df[df['is_english']]
    # Drop the 'is_english' helper column
    df_english = df_english.drop(columns=['is_english'])
    dropped_count = df.shape[0] - df_english.shape[0]
    print(f"Number of rows dropped (not in English): {dropped_count}")
    return df_english


def drop_na(df, col_name):
    """
    Drops rows with missing values in a specified column.

    Parameters:
        df (pandas.DataFrame): The DataFrame to clean.
        col_name (str): The name of the column to check for missing values.

    Returns:
        pandas.DataFrame: A DataFrame with rows dropped where the specified column had missing values.
    """
    df_cleaned = df.dropna(subset=[col_name])
    dropped_count = df.shape[0] - df_cleaned.shape[0]
    print(f"Number of rows dropped due to missing {col_name}: {dropped_count}")
    return df_cleaned


def keep_recent_abstracts(df):
    """
    Retains only the most recent abstracts for each professor, based on the 'Publication Year'.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing columns for 'Professor', 'Authors', and 'Publication Year'.

    Returns:
        pandas.DataFrame: A DataFrame with only the most recent abstracts for each professor.
    """
    df_sorted = df.sort_values(by=['Professor', 'Publication Year'], ascending=[True, False])
    df_final = df_sorted.drop_duplicates(subset=['Professor', 'Authors'], keep='first')
    dropped_count = df_sorted.shape[0] - df_final.shape[0]
    print(f"Number of rows dropped due to duplicate abstracts: {dropped_count}")
    return df_final


def clean_data(df):
    """
    Cleans the input DataFrame by applying various data cleaning steps, such as dropping irrelevant columns,
    filtering for English abstracts, handling missing values, and retaining only the most recent abstracts.

    Parameters:
        df (pandas.DataFrame): The DataFrame to be cleaned.

    Returns:
        pandas.DataFrame: The cleaned DataFrame after applying all the cleaning steps.
    """
    # Drop the columns - Title and Total Citations
    print("Cleaning the data...")
    df = df.drop(columns=['Title', 'Total Citations'])
    df = keep_only_english(df)
    df = drop_na(df, 'Publication Year')
    df = keep_recent_abstracts(df)
    print("Cleaning done")
    return df


if __name__ == '__main__':
    """
    Main script to load the raw dataset, clean it using the defined cleaning steps,
    and save the cleaned dataset as a CSV file.
    """
    print("Loading the data...")
    df = pd.read_excel('../data/scrapped_dataset.xlsx')
    df = clean_data(df)
    print("Saving the df as csv file...")
    df.to_csv('../data/self_cleaned_dataset.csv', index=False)
    print("Saved the cleaned dataset(self_cleaned_dataset.csv) is under folder 'data' with rows:", df.shape[0])
