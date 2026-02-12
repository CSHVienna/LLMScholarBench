import requests
import json
import pandas as pd

URI = 'https://openalex.org/I'

def fetch_openalex_data(file_path, output_file=None):
    """
    Fetch JSON data from the OpenAlex REST API for a list of IDs.

    Parameters:
        file_path (str): Path to the file containing a list of IDs (one per line).
        output_file (str): Optional path to save the JSON data as a file.

    Returns:
        dict: A dictionary where keys are IDs and values are the corresponding JSON objects.
    """
    base_url = "https://api.openalex.org/institutions"

    # Read the IDs from the file
    with open(file_path, 'r') as file:
        ids = [line.strip() for line in file if line.strip()]

    results = {}
    final_results = []

    for id_ in ids:
        try:
            # Make a GET request to the OpenAlex API for each ID
            response = requests.get(f"{base_url}?filter=openalex:I{id_}")
            response.raise_for_status()

            # Parse the JSON response
            data = response.json()
            institutions = data.get('results', [])
            institution_data = None 

            for institution in institutions:

                if institution.get('id').replace(URI,'') != id_:
                    continue

                # id_institution,cited_by_count,country_code,created_date,updated_date,display_name,
                # display_name_acronyms,ror,2yr_mean_citedness,h_index,i10_index,type,works_count
                
                summary_stats = institution.get('summary_stats', {})
                geo = institution.get('geo', {})

                # Store the result
                institution_data = {
                        'id_institution': id_,
                        'cited_by_count': institution.get('cited_by_count', None),
                        'country_code': institution.get('country_code', None),
                        'city': geo.get('country_code', None),
                        'created_date': institution.get('created_date', None),
                        'updated_date': institution.get('updated_date', None),
                        'display_name': institution.get('display_name', None),
                        'display_name_acronyms': ";".join(institution.get('display_name_acronyms', [])),
                        'ror': institution.get('ror', '').split('/')[-1] if institution.get('ror') else None,
                        '2yr_mean_citedness': summary_stats.get('2yr_mean_citedness', None),
                        'h_index': summary_stats.get('h_index', None),
                        'i10_index': summary_stats.get('i10_index', None),
                        'type': institution.get('type', None),
                        'works_count': institution.get('works_count', None)}
                
                print(institution_data['display_name'])
                break

            if institution_data is not None:
                final_results.append(institution_data)
            else:
                print(f"Data not found for ID {id_}")
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for ID {id_}: {e}")

    print(f'{len(final_results)} insitutions fetched.')
    df = pd.DataFrame(final_results)
    print(df.head(10))
    df.to_csv(output_file, index=False)
    
    return final_results

# Example usage
# Assuming `ids.txt` contains the list of IDs (one per line)
file_path = "../data/temp/id_institution_oa_missing.txt"
output_file = "../data/temp/instutions_missing.csv"

# Fetch data and save to `results.json`
data = fetch_openalex_data(file_path, output_file)
print("Data fetching completed.")
print(len(data))
