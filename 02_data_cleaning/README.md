# Data Cleaning

data_cleaning.py - This Python script performs data cleaning on a dataset of research abstracts, focusing on filtering rows based on language, removing duplicates, and retaining only recent records for each professor having same co-authors. The script is designed to help prepare the dataset for further analysis and exploration.

## Files

- `data_cleaning.py`: The main script containing data cleaning functions.
- `scraped_dataset.xlsx`: The input dataset containing research abstracts, professor names, publication years, etc.

## Output

- 'cleaned_dataset.csv' : A cleaned csv file with irrelevant columns dropped, non-English abstracts removed, missing publication years handled, and only recent abstracts retained for each professor.


## Usage

The script can be executed directly to clean the dataset. The input file (`scraped_dataset.xlsx`) should be placed in the `../data/` directory relative to the script.

1. **Input**: The input data must be in Excel format with the following columns:
   - `Abstract`: The abstract text to be filtered for language.
   - `Professor`: The name of the professor associated with the abstract.
   - `Authors`: The authors of the research publication.
   - `Publication Year`: The year the publication was released.
   - `Title`: The title of the publication.
   - `Total Citations`: The number of citations for the publication.

2. **Cleaning Steps**:
   - **Drop Columns**: Removes the `Title` and `Total Citations` columns.
   - **Language Filtering**: Only retains rows where the `Abstract` is in English using the `langdetect` library.
   - **Handle Missing Values**: Removes rows where the `Publication Year` is missing.
   - **Keep Recent Abstracts**: For each professor, only keeps the most recent abstract.

## Functions
The script contains the following key functions:

### `keep_only_english(df)`
- Filters the DataFrame to retain only rows where the 'Abstract' column is in English.
- Outputs a DataFrame containing only English abstracts and prints the number of rows dropped due to language detection.
### `drop_na(df, col_name)`
- Drops rows with missing values in a specified column.
- Outputs a cleaned DataFrame and prints the number of rows dropped due to missing values.
### `keep_recent_abstracts(df)`
- Retains only the most recent abstracts for each professor based on the 'Publication Year' and 'Authors' columns.
- Outputs a DataFrame containing only the most recent abstracts and prints the number of rows dropped due to duplicates.
### `clean_data(df)`
- Integrates the above functions to clean the input DataFrame.
- Drops the 'Title' and 'Total Citations' columns, filters for English abstracts, removes rows with missing publication years, and retains the most recent abstracts.

