# Evaluator

This part of the project evaluates LLM responses in physics literature, consisting of two main components:

## Enhancer
Processes LLM outputs and enriches them with additional data:
- Validates author names against OpenAlex and APS datasets
- Adds metadata (Nobel prizes, gender, publication years)
- Checks publication fields and authorship
- Handles various data formats

## Analyser
Creates tables and statistics from the enhanced data:
- Error rates
- Consistency measures
- Factuality checks
- Similarity metrics
- Rankings
- Demographic analysis
