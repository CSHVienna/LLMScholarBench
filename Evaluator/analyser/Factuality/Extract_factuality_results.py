import os
import json
import re

# Function to check if two year ranges overlap
def years_overlap(year_range_1, year_start_2, year_end_2):
    """Check if year_range_1 overlaps with the second range [year_start_2, year_end_2]."""
    ymin1, ymax1 = year_range_1
    return ymax1 >= year_start_2 and ymin1 <= year_end_2


def parse_career_age(career_age_str):
    """Parse and clean the 'Career Age' string into a numeric value for comparison."""
    career_age_str = career_age_str.strip().lower()

    # Remove non-numeric words like 'approximately', 'over', 'about', etc.
    career_age_str = re.sub(r'\b(approximately|about|over)\b', '', career_age_str).strip()
    
    # Handle just a plain number (like "126")
    if re.match(r'^\d+$', career_age_str):
        return int(career_age_str)

    # Handle common formats like "60+", "4 years", "more than 10 years", etc.
    if re.match(r'^\d+\+', career_age_str):  # Example: "60+"
        return int(re.findall(r'^\d+', career_age_str)[0])
    
    elif re.match(r'^\d+-\d+', career_age_str):  # Example: "10-15 years"
        range_str = career_age_str.split('-')[1]  # Take the upper bound in a range
        return int(re.findall(r'\d+', range_str)[0])
    
    elif re.match(r'^\d+ years?', career_age_str):  # Example: "4 years"
        return int(re.findall(r'\d+', career_age_str)[0])

    # Handle formats like "~10 years" or "~10"
    elif re.match(r'^~\d+', career_age_str):  # Example: "~10 years" or "~10"
        return int(re.findall(r'\d+', career_age_str)[0])

    # Handle formats like "more than 10 years"
    elif re.match(r'more than \d+', career_age_str):
        return int(re.findall(r'\d+', career_age_str)[0]) + 1

    return None


def process_directory(root_directory):
    # The final dictionary to store results
    results = {}

    # Traverse the directory structure
    for config_folder in os.listdir(root_directory):
        config_path = os.path.join(root_directory, config_folder)
        if os.path.isdir(config_path):
            results[config_folder] = {}  # Initialize dict for each config
            
            for run_folder in os.listdir(config_path):
                run_path = os.path.join(config_path, run_folder)
                if os.path.isdir(run_path):
                    for use_case_folder in os.listdir(run_path):
                        use_case_path = os.path.join(run_path, use_case_folder)
                        if os.path.isdir(use_case_path):
                            # Initialize for each use case
                            if use_case_folder not in results[config_folder]:
                                # Initialize lists to store values for each run
                                results[config_folder][use_case_folder] = {
                                    "count_names": [],
                                    "count_in_oa": [],
                                    "count_in_aps": [],
                                    "author_correct_field": []  # New for field_ use cases
                                }

                                # Add fields for 'field_' use cases
                                if use_case_folder.startswith('field_'):
                                    results[config_folder][use_case_folder].update({
                                        "unique_dois": [],
                                        "total_dois": [],
                                        "count_in_openalex_pub": [],
                                        "count_aps_publication": [],
                                        "count_correct_authorship": [],
                                        "count_correct_field": []  # New for publications in field_ use cases
                                    })

                                # Add fields for 'epoch_' use cases
                                if use_case_folder.startswith('epoch_'):
                                    results[config_folder][use_case_folder].update({
                                        "count_correct_author_epoch": [],
                                        "count_correct_recommended_epoch": []
                                    })

                                # Add fields for 'seniority_' use cases
                                if use_case_folder.startswith('seniority_'):
                                    results[config_folder][use_case_folder].update({
                                        "count_correct_author_seniority": [],
                                        #"count_correct_recommended_career_age": []
                                        "mean_career_age_error": []
                                    })

                            # Initialize counters for the current run
                            count_names = 0
                            count_in_oa = 0
                            count_in_aps = 0
                            author_correct_field = 0  # For field_ use case
                            count_correct_authorship = 0
                            unique_dois_set = set()
                            total_dois = 0
                            count_in_openalex_pub = 0
                            count_aps_publication = 0
                            count_correct_field = 0  # For field_ publications
                            count_correct_author_epoch = 0
                            count_correct_recommended_epoch = 0
                            count_correct_author_seniority = 0
                            count_correct_recommended_career_age = 0

                            for file_name in os.listdir(use_case_path):
                                if file_name.startswith('validation_result_') and file_name.endswith('.json'):
                                    file_path = os.path.join(use_case_path, file_name)
                                    with open(file_path, 'r') as f:
                                        data = json.load(f)

                                        # Process enhanced_authors
                                        if "enhanced_authors" in data:
                                            enhanced_authors = data["enhanced_authors"]
                                            if enhanced_authors:
                                                count_names = len(enhanced_authors)
                                                count_in_oa = sum(1 for author in enhanced_authors if author.get("is_in_openalex", False))
                                                count_in_aps = sum(1 for author in enhanced_authors if author.get("has_published_in_aps", False) or author.get("any_candidate_in_aps", False))
                                                # Count correct_authorship if use case starts with 'field_'
                                                if use_case_folder.startswith('field_'):
                                                    author_correct_field = sum(1 for author in enhanced_authors if author.get("correct_authorship", False))

                                                # Handle 'epoch_' use cases
                                                if use_case_folder.startswith('epoch_'):
                                                    # Determine the epoch range based on the folder name
                                                    if '1950s' in use_case_folder:
                                                        epoch_start, epoch_end = 1950, 1960
                                                    elif '2000s' in use_case_folder:
                                                        epoch_start, epoch_end = 2000, 2010
                                                    else:
                                                        epoch_start, epoch_end = None, None
                                                    
                                                    # Count authors whose years_of_activity overlap with the epoch
                                                    if epoch_start and epoch_end:
                                                        count_correct_author_epoch = sum(
                                                            1 for author in enhanced_authors 
                                                            if "years_of_activity" in author 
                                                            and years_overlap(author["years_of_activity"], epoch_start, epoch_end)
                                                        )

                                                        # # Count authors whose "Years" overlap with the epoch
                                                        # count_correct_recommended_epoch = sum(
                                                        #     1 for author in enhanced_authors 
                                                        #     if "Years" in author 
                                                        #     and years_overlap([int(y) for y in author["Years"].split('-')], epoch_start, epoch_end)
                                                        # )

                                                    
                                                        # Count authors whose "Years" overlap with their years_of_activity
                                                        count_correct_recommended_epoch = sum(
                                                            1 for author in enhanced_authors 
                                                            if "Years" in author and "years_of_activity" in author
                                                            and years_overlap([int(y) for y in author["Years"].split('-')], author["years_of_activity"][0], author["years_of_activity"][1])
                                                        )

                                                # Handle 'seniority_' use cases
                                                if use_case_folder.startswith('seniority_'):
                                                    if 'early_career' in use_case_folder:
                                                        age_limit = 10  # Early career: <= 10 years
                                                    elif 'senior' in use_case_folder:
                                                        age_limit = 20  # Senior: >= 20 years
                                                    else:
                                                        age_limit = None
                                                    
                                                    # Count correct_author_seniority based on academic_age
                                                    if age_limit:
                                                        if 'early_career' in use_case_folder:
                                                            count_correct_author_seniority = sum(
                                                                1 for author in enhanced_authors
                                                                if "academic_age" in author and author["academic_age"] <= age_limit
                                                            )
                                                        elif 'senior' in use_case_folder:
                                                            count_correct_author_seniority = sum(
                                                                1 for author in enhanced_authors
                                                                if "academic_age" in author and author["academic_age"] >= age_limit
                                                            )

                                                        # # Count correct_recommended_career_age based on Career Age
                                                        # count_correct_recommended_career_age = sum(
                                                        #     1 for author in enhanced_authors
                                                        #     if "Career Age" in author and (parsed_age := parse_career_age(author["Career Age"])) is not None
                                                        #     and ((parsed_age <= 10 and 'early_career' in use_case_folder) or (parsed_age >= 20 and 'senior' in use_case_folder))
                                                        # )
                                                        # Initialize an empty list to store the errors for each author
                                                        errors = []

                                                        # Calculate the errors between Career Age and Academic Age
                                                        for author in enhanced_authors:
                                                            if "Career Age" in author and "academic_age" in author:
                                                                parsed_career_age = parse_career_age(author["Career Age"])
                                                                if parsed_career_age is not None:
                                                                    # Calculate the error (difference) between Career Age and Academic Age
                                                                    error = parsed_career_age - author["academic_age"]
                                                                    errors.append(error)  # Store the error in the list

                                                        # Now calculate the mean error if there are any errors
                                                        if errors:
                                                            mean_error = sum(errors) / len(errors)
                                                        else:
                                                            mean_error = None  # No valid errors to calculate the mean

                                        if use_case_folder.startswith('field_') and "enhanced_publications" in data:
                                            enhanced_publications = data["enhanced_publications"]
                                            for pub in enhanced_publications:
                                                # Process DOI for both unique DOIs and total DOIs
                                                doi = pub.get("DOI")
                                                if doi:
                                                    unique_dois_set.add(doi)
                                                    total_dois += 1
                                                
                                                # Count if the publication is in OpenAlex
                                                if pub.get("is_in_openalex", False):
                                                    count_in_openalex_pub += 1
                                                
                                                # Count if the publication is an APS publication
                                                if pub.get("is_aps_publication", False):
                                                    count_aps_publication += 1
                                                
                                                # Count correct_authorship in publications
                                                if pub.get("correct_authorship", False):
                                                    count_correct_authorship += 1
                                                
                                                # Count correct_field in publications
                                                if pub.get("correct_field", False):
                                                    count_correct_field += 1

                            # Append the counts for this run
                            results[config_folder][use_case_folder]["count_names"].append(count_names)
                            results[config_folder][use_case_folder]["count_in_oa"].append(count_in_oa)
                            results[config_folder][use_case_folder]["count_in_aps"].append(count_in_aps)
                            
                            # Append author correct field for field_ use cases
                            if use_case_folder.startswith('field_'):
                                results[config_folder][use_case_folder]["author_correct_field"].append(author_correct_field)
                                results[config_folder][use_case_folder]["count_correct_authorship"].append(count_correct_authorship)
                                results[config_folder][use_case_folder]["unique_dois"].append(len(unique_dois_set))  # Append unique DOI count
                                results[config_folder][use_case_folder]["total_dois"].append(total_dois)  # Append total DOI count
                                results[config_folder][use_case_folder]["count_in_openalex_pub"].append(count_in_openalex_pub)  # Append OpenAlex pub count
                                results[config_folder][use_case_folder]["count_aps_publication"].append(count_aps_publication)  # Append APS pub count
                                results[config_folder][use_case_folder]["count_correct_field"].append(count_correct_field)  # Append correct field count

                            # For 'epoch_' use cases, append the additional lists
                            if use_case_folder.startswith('epoch_'):
                                results[config_folder][use_case_folder]["count_correct_author_epoch"].append(count_correct_author_epoch)
                                results[config_folder][use_case_folder]["count_correct_recommended_epoch"].append(count_correct_recommended_epoch)

                            # For 'seniority_' use cases, append the additional lists
                            if use_case_folder.startswith('seniority_'):
                                results[config_folder][use_case_folder]["count_correct_author_seniority"].append(count_correct_author_seniority)
                                #results[config_folder][use_case_folder]["count_correct_recommended_career_age"].append(count_correct_recommended_career_age)
                                results[config_folder][use_case_folder]["mean_career_age_error"].append(mean_error)

    # Save results as JSON
    output_file = os.path.join(root_directory, 'descriptive_statistics_by_run.json')
    with open(output_file, 'w') as outfile:
        json.dump(results, outfile, indent=4)

    print(f"Descriptive statistics saved to {output_file}")

# Example of how to call this function with your directory
root_directory = "././experiments_validation_results/updated_organized_results"
process_directory(root_directory)

