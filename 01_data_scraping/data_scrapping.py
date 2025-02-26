import csv
from scholarly import scholarly
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
import pickle
import requests
import re

# API key for Semantic Scholar
#API_KEY = "..."
# Semantic Scholar's URL
BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

headers = {
    "x-api-key": "8ysEwCCtwB7NXT5qsrWl85INQh9pnXB96tsPGfPH"
}

def save_progress_to_pickle(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_progress_from_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []

def process_professor(professor_name):
    # Retrieves the list of publications for a professor from Google Scholar.
    # Extracts the title, publication year, authors, and citation count for each publication.
    # Outputs a list of publication details for the professor.

    results = []
    print(f"Processing {professor_name}...")
    try:
        # search Professor
        search_query = scholarly.search_author(professor_name)
        author = next(search_query)
        scholarly.fill(author)  # get details

        publications = author.get('publications', [])

        if not publications:
            print(f"No publications found for Professor: {professor_name}")
            return results

        # Define a function to fill in the paper information and add a random delay
        def fill_publication(pub):
            scholarly.fill(pub)
            time.sleep(random.uniform(0.5, 1.5))
            return pub

        # Use ThreadPoolExecutor to fill paper information in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_pub = {executor.submit(fill_publication, pub): pub for pub in publications}
            for future in as_completed(future_to_pub):
                pub = future_to_pub[future]
                try:
                    pub = future.result()

                    title = pub.get('bib', {}).get('title')
                    if not title:
                        print(f"Title missing for a publication of Professor: {professor_name}, skipping...")
                        continue  # Skip papers without titles

                    # get other information, if not exists, use default value
                    pub_year = pub.get('bib', {}).get('pub_year', 'No year')
                    authors = pub.get('bib', {}).get('author', 'No authors')
                    num_citations = pub.get('num_citations', 'N/A')
                    
                    results.append([professor_name, title, pub_year, authors, num_citations])
                except Exception as e:
                    print(f"Error processing publication: {e}")
                    title = pub.get('bib', {}).get('title', 'No title')
                    results.append([professor_name, title, 'No year', 'No authors', 'N/A'])
    except StopIteration:
        print(f"No author found for Professor: {professor_name}")
    except Exception as e:
        print(f"Error processing Professor {professor_name}: {e}")
    return results

def clean_title(title):
    # Define a function to clean up the title, remove special characters and subtitles
    return title.split(':')[0].strip().lower()

def clean_text(text):
    return re.sub(r'[^\x20-\x7E]+', '', text)

def get_abstracts_for_titles(titles):
    # Searches for abstracts in the Semantic Scholar API based on a list of article titles.
    # Saves the abstract information to the specified Excel file.
    
    results = {}

    # Set limit of 1 request per second
    RATE_LIMIT = 1
    DELAY = 1.0 / RATE_LIMIT

    # search the abstracts
    for idx, title in enumerate(titles):
        clean_title_str = clean_title(title)

        params = {
            'query': clean_title_str,
            'fields': 'title,abstract',
            'limit': 3
        }

        try:
            response = requests.get(BASE_URL, headers=headers, params=params)
            data = response.json()

            if data.get('data') and len(data['data']) > 0:
                paper = data['data'][0]  # Get the first matching result
                abstract = paper.get('abstract', 'N/A')

                if not isinstance(abstract, str):
                    abstract = 'N/A'

                abstract = clean_text(abstract)
            else:
                abstract = 'N/A'

        except requests.exceptions.RequestException as e:
            print(f"Error occurred while retrieving abstract for '{clean_title_str}': {e}")
            abstract = 'N/A'

        results[title] = abstract
        time.sleep(DELAY)

    return results


if __name__ == '__main__':
    
    file_path = './data/EngineeringFaculty-small.csv'
    df = pd.read_csv(file_path)

    professor_names = df['Name'].tolist()

    # save to local file
    csv_file_path = './data/professor_papers.csv'
    pickle_file_path = './data/professor_results.pkl'

    # load data from pickle file
    professor_results = load_progress_from_pickle(pickle_file_path)

    # Retrieve professor publications
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Professor', 'Title', 'Publication Year', 'Authors', 'Total Citations'])  # CSV Header

        # Using ThreadPoolExecutor to process the professor list in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_professor = {executor.submit(process_professor, name): name for name in professor_names}
            for future in as_completed(future_to_professor):
                professor_name = future_to_professor[future]
                try:
                    professor_results = future.result()
                    writer.writerows(professor_results)
                    
                    save_progress_to_pickle(pickle_file_path, professor_results)
                except Exception as e:
                    print(f"Error processing Professor {professor_name}: {e}")

    # Load the professor papers into a DataFrame
    papers_df = pd.read_csv(csv_file_path)
    titles = papers_df['Title'].tolist()

    # Retrieve abstracts for each title
    abstract_dict = get_abstracts_for_titles(titles)

    # Merge the abstracts into the DataFrame
    papers_df['Abstract'] = papers_df['Title'].map(abstract_dict)

    # Save the final combined dataset
    output_file_path = './data/scraped_dataset.xlsx'
    papers_df.to_excel(output_file_path, index=False)
    print(f"The combined dataset has been saved to {output_file_path}")
