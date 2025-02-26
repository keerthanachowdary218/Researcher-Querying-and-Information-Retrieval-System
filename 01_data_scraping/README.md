# Data Scrapping
We used google scholar to get basic informations, such as paper titles, publication years, co-authors, and total citations.
And then we used the APIs of Semantic Scholar to retrieve the abstract of each paper.
In addition, we manually added some paper information from ScienceDirect that was not scraping from code.

## Files

- `data_scrapping.py`: The main script for processing publications and retrieving abstracts.
- `EngineeringFaculty-small.csv`: Input dataset containing professor names, used to retrieve paper title、publication year etc.

## Output

- `scraped_dataset.xlsx: A combined Excel file that contains each professor’s publication information, including the title, publication year, authors, citation count, and abstract for each paper.

## Usage

The script can be executed directly. Input files should be placed in the appropriate paths as defined in the code.

1. **Input**:
   - EngineeringFaculty-small.csv: Contains a list of professor names, used to retrieve their publications from Google Scholar.

2. **Abstract Retrieval**:
   - The script uses the Semantic Scholar API to retrieve abstracts for each article title retrieved from Google Scholar and stores the results in scraped_dataset.xlsx.
  
## Processing Steps

1. Process Professor Papers:
   - Retrieves publication details from Google Scholar for each professor.
   - Uses ThreadPoolExecutor to process multiple professors concurrently.
   - Progress is periodically saved to a local file (pickle) to prevent data loss.
     
2. Retrieve Abstracts:
   - For each article title, fetches the corresponding abstract using the Semantic Scholar API.
   - The script limits the rate of requests to one per second to avoid API throttling.
   - Merges the abstracts into the professor publication details and saves the final result in scraped_dataset.xlsx.

## Functions

### `process_professor(professor_name)`
- Retrieves the list of publications for a professor from Google Scholar.
- Extracts the title, publication year, authors, and citation count for each publication.
- Outputs a list of publication details for the professor.

### `get_abstract(titles, output_file_path)`
- Searches for abstracts in the Semantic Scholar API based on a list of article titles.
- Saves the abstract information to the specified Excel file.

### `save_results_to_csv(file_path, data)`
- Saves the professor’s publication details to a CSV file.

### `save_progress_to_pickle(file_path, data)`
- Saves the current processing state in a pickle file to allow progress recovery in case of interruption.

### `load_progress_from_pickle(file_path)`
- Loads the previously saved progress from a pickle file to resume processing where it was left off.











