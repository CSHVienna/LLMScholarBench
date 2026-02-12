# OpenAlex Data Retrieval

This project contains scripts to retrieve publication, author, and institution data from the OpenAlex API.

## Overview

The project consists of five main scripts that form a data retrieval pipeline:

1. `scripts/openalex_api_dois.py`: Retrieves data for publications based on their DOIs.
2. `scripts/extract_author_ids.py`: Extracts unique author IDs from the publication data.
3. `scripts/openalex_api_authors.py`: Retrieves data for authors based on the extracted author IDs.
4. `scripts/extract_institution_ids.py`: Extracts unique institution IDs from the author data.
5. `scripts/openalex_api_institutions.py`: Retrieves data for institutions based on the extracted institution IDs.

## Data Source

The initial data used in our research, specifically the APS dataset, can be requested through the official APS website (https://journals.aps.org/datasets). This dataset provides the starting point for our OpenAlex data retrieval pipeline.

## Project Structure

```
project_root/
│
├── data/
│   ├── input/
│   │   └── publications.csv
│   │
│   └── output/
│       ├── openalex_dois_data.json
│       ├── unique_author_ids.txt
│       ├── openalex_authors_data.json
│       ├── unique_institution_ids.txt
│       └── openalex_institutions_data.json
│
├── scripts/
│   ├── openalex_api_dois.py
│   ├── extract_author_ids.py
│   ├── openalex_api_authors.py
│   ├── extract_institution_ids.py
│   └── openalex_api_institutions.py
│
├── README.md
├── requirements.txt
└── structure.txt
```

## Requirements

- Python 3.7+
- Required Python packages are listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

## Usage

The scripts should be run in the following order to create the full data pipeline:

### 1. Retrieving Publication Data

1. Ensure your `publications.csv` file is in the `data/input/` directory.
2. Run the `openalex_api_dois.py` script:

   ```
   python scripts/openalex_api_dois.py
   ```

   This script reads from `data/input/publications.csv`, which should contain at least a 'doi' column. It retrieves data for each DOI and saves the results in `data/output/openalex_dois_data.json`.

### 2. Extracting Author IDs

Run the `extract_author_ids.py` script:

```
python scripts/extract_author_ids.py
```

This script reads from `data/output/openalex_dois_data.json` and extracts unique author IDs, saving them to `data/output/unique_author_ids.txt`.

### 3. Retrieving Author Data

Run the `openalex_api_authors.py` script:

```
python scripts/openalex_api_authors.py
```

This script reads the author IDs from `data/output/unique_author_ids.txt`, retrieves data for each author, and saves the results in `data/output/openalex_authors_data.json`.

### 4. Extracting Institution IDs

Run the `extract_institution_ids.py` script:

```
python scripts/extract_institution_ids.py
```

This script reads from `data/output/openalex_authors_data.json` and extracts unique institution IDs, saving them to `data/output/unique_institution_ids.txt`.

### 5. Retrieving Institution Data

Run the `openalex_api_institutions.py` script:

```
python scripts/openalex_api_institutions.py
```

This script reads the institution IDs from `data/output/unique_institution_ids.txt`, retrieves data for each institution, and saves the results in `data/output/openalex_institutions_data.json`.

## Data Retrieved

### Publication Data

- DOI
- Primary topic
- Authors
- Title
- Cited by count
- Language
- Publication date
- Referenced works
- Error (if any)

### Author Data

- Author ID
- Affiliations
- Cited by count
- Counts by year
- Created date
- Display name
- Display name alternatives
- Summary stats
- Updated date
- Works count
- X concepts

### Institution Data

- Institution ID
- Display name
- Works count
- Cited by count
- Summary stats

## Sample Size

The API scripts (`openalex_api_dois.py`, `openalex_api_authors.py`, and `openalex_api_institutions.py`) include a `sample_size` parameter in the `if __name__ == "__main__":` block. Set this to an integer value to process only a subset of the data, or to `None` to process all data.

## Note

These scripts include rate limiting (sleeping for 0.1 seconds between requests) to avoid overwhelming the OpenAlex API. Adjust this as needed based on your use case and the API's rate limits.