# Technical Skills Search System

A comprehensive skill search system that processes employee data to provide intelligent skill-based search capabilities using advanced language metrics and machine learning techniques.

## Overview

This system processes a dataset of approximately 2,500 employees with their associated skills and roles to create an intelligent technical skill search engine. The search results are ranked by relevance using a combination of:

- **TF-IDF (Term Frequency-Inverse Document Frequency)** - Measures skill importance/uniqueness
- **Probability Metrics** - Calculates skill frequency across the dataset
- **Levenshtein Distance** - Handles fuzzy matching for partial queries

## System Architecture

The system is divided into three main phases:

### Phase 1: Data Processing & Organization

**Objective**: Extract, organize, and structure employee skills data by role.

**Key Operations**:
- Utilizes Pandas to extract skills and roles from employee entries
- Treats roles as index columns for data segmentation
- Converts skills data into structured CSV files organized by role
- Creates data structures for subsequent processing phases

**Output Files**:
- `csv_docs/` folder containing individual `<role>.csv` files
- `pickle_objs/` folder with binary files for efficient data retrieval:
 - `Roles.pkl` - Set of unique roles in the dataset
 - `Tech_Skills_List.pkl` - Unified list of all technical skills (with duplicates)
 - `Tech_Skills_Vocab.pkl` - Vocabulary of unique technical skills
 - `ROWS.pkl` - 2D array for TF-IDF processing

**Data Structure**:
- Individual CSV files contain separate entries for each employee's skillset
- Role-based lists aggregate all skills for employees within each role
- 2D array structure prepared for TF-IDF vectorization

### Phase 2: Metric Calculation

**Objective**: Calculate relevance metrics for intelligent search ranking.

#### TF-IDF Calculation
- Treats each role as a separate document
- Uses scikit-learn's TFIDFVectorizer to generate TF-IDF matrix
- Calculates average TF-IDF across all roles for each skill
- Normalizes by the number of roles containing each skill
- Outputs: `Avg_Tfidf.pkl` - Sorted dictionary (skills → TF-IDF values)

#### Probability Calculation
- Calculates skill probability: `count(skill) / total_skills`
- Measures frequency of each skill across the entire dataset
- Outputs: `Probability.pkl` - Sorted dictionary (skills → probability values)

### Phase 3: Search Implementation

**Objective**: Implement intelligent search with multi-layered relevance ranking.

#### Primary Layer: TF-IDF Matching
- Accepts user input (3+ characters)
- Matches skills with highest TF-IDF values first
- Creates `Selected_Words` list of matching high-relevance skills
- Handles partial matches through prefix matching

#### Secondary Layer: Levenshtein Distance
- Activates when fewer than 4 primary matches found
- Computes edit distance between input and remaining skills
- Sorts by ascending distance (closest matches first)
- Appends best matches to `Selected_Words`

#### Tertiary Layer: Probability + Distance
- Processes high-probability skills sequentially
- Calculates Levenshtein distance against selected words
- Sorts and appends to final output list `Tech_list`
- Combines probability weighting with distance metrics

## Features

- **Multi-layered Relevance**: Combines TF-IDF, probability, and edit distance for comprehensive ranking
- **Fuzzy Matching**: Handles typos and partial queries through Levenshtein distance
- **Role-based Organization**: Maintains skill-role associations for targeted searches
- **Efficient Storage**: Uses pickle files for fast data retrieval
- **Scalable Architecture**: Designed to handle large employee datasets

## Dependencies

- **pandas** - Data manipulation and analysis
- **scikit-learn** - TF-IDF vectorization
- **enchant** - Levenshtein distance calculation
- **pickle** - Binary data serialization

## Usage

1. **Data Input**: Provide employee dataset with skills and roles
2. **Processing**: Run Phase 1 to organize data, Phase 2 to calculate metrics
3. **Search**: Use Phase 3 to perform intelligent skill searches
4. **Results**: Receive ranked skill matches based on relevance metrics

## Output Structure

Search results are ranked by:
1. **Primary**: TF-IDF relevance (importance/uniqueness)
2. **Secondary**: Levenshtein distance (similarity to query)
3. **Tertiary**: Probability combined with distance metrics

This multi-layered approach ensures that search results are both relevant and comprehensive, handling both exact matches and approximate queries effectively.
