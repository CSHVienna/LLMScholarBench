import os
import json

def load_authors(file_path):
    with open(file_path, 'r') as f:
        return {json.loads(line)['openalex_id']: json.loads(line)['id_author'] for line in f}

def load_aps_author_stats(file_path):
    with open(file_path, 'r') as f:
        return {json.loads(line)['id_author']: json.loads(line) for line in f}

def get_decade(year):
    return (year // 10) * 10

def update_enhanced_authors(file_path, authors, aps_author_stats):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if 'enhanced_authors' in data and isinstance(data['enhanced_authors'], list):
        for author_info in data['enhanced_authors']:
            if 'key' in author_info and author_info['key'] in authors:
                # Get id_author
                id_author = authors[author_info['key']]
                author_info['id_author'] = id_author
                
                # Get decade of first publication and year range
                if id_author in aps_author_stats:
                    aps_years = aps_author_stats[id_author]['aps_years_of_activity']
                    if aps_years and isinstance(aps_years, list):
                        valid_years = [int(y) for y in aps_years if str(y).isdigit()]
                        if valid_years:
                            first_year = min(valid_years)
                            last_year = max(valid_years)
                            author_info['decade_first_publication'] = get_decade(first_year)
                            author_info['year_range'] = f"{first_year}-{last_year}"
                        else:
                            author_info['decade_first_publication'] = 'unknown'
                            author_info['year_range'] = 'unknown'
                    else:
                        author_info['decade_first_publication'] = 'unknown'
                        author_info['year_range'] = 'unknown'
                else:
                    author_info['decade_first_publication'] = 'not in aps'
                    author_info['year_range'] = 'not in aps'
            else:
                author_info['id_author'] = 'not in aps'
                author_info['decade_first_publication'] = 'not in aps'
                author_info['year_range'] = 'not in aps'
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Updated {file_path}")
    else:
        print(f"No 'enhanced_authors' list found in {file_path}")

def process_directory(directory_path, authors, aps_author_stats):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.json') and file.startswith('validation_result'):
                file_path = os.path.join(root, file)
                update_enhanced_authors(file_path, authors, aps_author_stats)

def main():

    authors_file = f"././organised_data/authors.json"
    aps_author_stats_file = f"././organised_data/aps_author_stats.json"
    results_dir = f"././experiments_validation_results/updated_organized_results"

    authors = load_authors(authors_file)
    aps_author_stats = load_aps_author_stats(aps_author_stats_file)

    print(f"Loaded {len(authors)} authors")
    print(f"Loaded {len(aps_author_stats)} APS author stats")

    process_directory(results_dir, authors, aps_author_stats)

if __name__ == "__main__":
    main()