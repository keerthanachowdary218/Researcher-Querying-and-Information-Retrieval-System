"""
In this module we will look at some of the general characteristics of
the dataset we are working with after pre-processing. This is a starting
point for the exploration of our data with the intention to spot any
anomalities in the data or any other features of interes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from collections import Counter

# Existing English stopwords from CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stopwords
from wordcloud import WordCloud, STOPWORDS

# Set a nice seaborn style
sns.set(style="whitegrid")


# Define custom stop words (additional to the ones included in python packages)
custom_stop_word_list = ["using", "via", "based", "model", "design", "impact",
                         "state", "efficient", "method", "study",
                         "©", "2023", "however,", "results", "high",
                         "use", "used", "show", "approach", "system",
                         "new", "two"]



def plot_top_authors(df, column='Professor', top_n=30, palette='Blues_d'):
    """
    Plots a bar chart of the top authors by publication count.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        column (str): The column name in the DataFrame representing the authors.
        top_n (int): The number of top authors to display (default is 30).
        palette (str): The color palette to use for the bar plot (default is 'Blues_d').

    Returns:
        None: Displays the plot.
    """
    # Count the number of papers by each author
    top_authors = df[column].value_counts().head(top_n)

    # Create a larger figure for better readability
    plt.figure(figsize=(10, 6))

    # Plot the top authors by publication count using seaborn's barplot
    sns.barplot(x=top_authors.index, y=top_authors.values, palette=palette)

    # Add title and labels
    plt.title(f"Top {top_n} Authors by Publication Count", fontsize=16)
    plt.xlabel("Authors", fontsize=14)
    plt.ylabel("Number of Publications", fontsize=14)

    # Rotate the x-axis labels for better readability if names are long
    plt.xticks(rotation=45, ha='right', fontsize=12)

    # Add a grid for better visual separation
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Show the plot with a tight layout
    plt.tight_layout()
    plt.show()


def plot_publications_over_time(df, year_column='Publication Year'):
    """
    Plots the number of publications per year.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        year_column (str): The column name in the DataFrame representing the publication years.

    Returns:
        None: Displays the plot.
    """
    # Filter rows where the year column is not null
    df_yeardist = df[df[year_column].notna()]

    # Convert the year column to numeric, coercing errors
    df_yeardist[year_column] = pd.to_numeric(df_yeardist[year_column], errors='coerce')

    # Count the number of papers published per year and sort by year
    publication_by_year = df_yeardist[year_column].value_counts().sort_index()

    # Create a larger figure for better readability
    plt.figure(figsize=(10, 6))

    # Plot the number of publications per year
    sns.lineplot(x=publication_by_year.index, y=publication_by_year.values)

    # Add title and labels
    plt.title("Publications Over Time", fontsize=16)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Number of Papers", fontsize=14)

    # Add a grid for better visual separation
    plt.grid(True, axis='both', linestyle='--', alpha=0.7)

    # Show the plot with a tight layout
    plt.tight_layout()
    plt.show()


def plot_most_common_words(df, text_column='Abstract', max_features=40):
    """
    Plots the most common words in a text column after removing stopwords.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        text_column (str): The column name in the DataFrame containing the text data.
        custom_stopwords (list): A list of custom stopwords to remove.
        max_features (int): The maximum number of features (words) to include (default is 40).

    Returns:
        None: Displays the plot.
    """
    custom_stopwords = list(sklearn_stopwords) + custom_stop_word_list

    # Create a count vectorizer with the custom stopwords
    vectorizer = CountVectorizer(stop_words=custom_stopwords, max_features=max_features)

    # Fit the vectorizer to the text column
    word_counts = vectorizer.fit_transform(df[text_column])

    # Create a DataFrame to display the most common words
    common_words = pd.DataFrame({
        'word': vectorizer.get_feature_names_out(),
        'count': word_counts.sum(axis=0).A1
    }).sort_values(by='count', ascending=False)

    # Plot the most common words using seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x='count', y='word', data=common_words, palette="Blues_d")

    # Add title and labels
    plt.title(f"Most Common Keywords in {text_column.capitalize()}", fontsize=16)
    plt.xlabel("Count", fontsize=14)
    plt.ylabel("Keywords", fontsize=14)

    # Add a grid for better visual separation
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Show the plot with a tight layout
    plt.tight_layout()
    plt.show()


def analyze_common_words_by_year(df, year_column='Publication Year', text_column='Abstract', target_year=None):
    """
    Analyzes the most common words in a text column by publication year, ignoring stopwords.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        year_column (str): The column name representing the publication year.
        text_column (str): The column name containing the text data (e.g., 'Abstract').
        custom_stopwords (list): A list of custom stopwords to remove (default is None).
        target_year (int): The year to display most common words for (default is None).

    Returns:
        dict: A dictionary of word frequencies by year.
        list: A list of the most common words in the target year (if specified).
    """
    # Download NLTK stopwords if not already installed
    nltk.download('stopwords', quiet=True)
    custom_stopwords = custom_stop_word_list

    # Use WordCloud's STOPWORDS set, combined with any custom stopwords
    stopwords = set(STOPWORDS)
    if custom_stopwords:
        stopwords.update(custom_stopwords)

    # Tokenize abstracts into words and remove stopwords
    df['abstract_tokens'] = df[text_column].apply(lambda x: [word.lower() for word in str(x).split() if word.lower() not in stopwords])

    # Group by year and collect all tokens
    grouped_by_year = df.groupby(year_column)['abstract_tokens'].sum()

    # Apply Counter to analyze frequent terms per year
    word_freq_by_year = {year: Counter(words) for year, words in grouped_by_year.items()}

    # If a target year is provided, return the most common words for that year
    most_common_words = None
    if target_year:
        most_common_words = word_freq_by_year.get(target_year, Counter()).most_common(10)
        print(f"Most common words in {target_year}: {most_common_words}")

    return word_freq_by_year, most_common_words


def plot_most_common_words_by_year(word_freq_by_year, year, top_n=20):
    """
    Plots the most common words for a specific year using Seaborn's barplot.

    Parameters:
        word_freq_by_year (dict): A dictionary containing word frequencies by year.
        year (int): The target year to analyze.
        top_n (int): The number of most common words to display (default is 20).

    Returns:
        None: Displays the plot.
    """
    # Get the most common words for the specified year
    most_common_words = word_freq_by_year.get(year, Counter()).most_common(top_n)

    # Unpack the words and their frequencies
    if most_common_words:
        words, frequencies = zip(*most_common_words)
        common_words_df = pd.DataFrame({'word': words, 'frequency': frequencies})
    else:
        print(f"No data available for the year {year}.")
        return

    # Create a bar plot for the most common words using seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x='frequency', y='word', data=common_words_df, palette="Blues_d")

    # Add labels and title
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Words', fontsize=12)
    plt.title(f"Most Common Words in {year}", fontsize=15)

    # Show the plot with a tight layout
    plt.tight_layout()
    plt.show()



def plot_wordcloud(df, text_column='Abstract', max_words=100, width=1600, height=800, background_color="white"):
    """
    Generates and plots a word cloud from a text column.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        text_column (str): The column containing the text to be visualized.
        max_words (int): The maximum number of words to include in the word cloud (default is 100).
        width (int): The width of the word cloud image (default is 1600).
        height (int): The height of the word cloud image (default is 800).
        background_color (str): The background color of the word cloud (default is "white").

    Returns:
        None: Displays the word cloud plot.
    """
    # Download NLTK stopwords if not already installed
    nltk.download('stopwords', quiet=True)
    custom_stopwords = custom_stop_word_list

    # Use WordCloud's STOPWORDS set, combined with any custom stopwords
    stopwords = set(STOPWORDS)
    if custom_stopwords:
        stopwords.update(custom_stopwords)
    # Combine all titles or abstracts into one large string
    all_text = ' '.join(df[text_column].dropna())

    # Generate the word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        stopwords=stopwords,
        max_words=max_words,
        background_color=background_color
    ).generate(all_text)

    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
