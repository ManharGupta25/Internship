The code takes in a dataset of ~2500 employees with their skills and roles.
It creates a technical skill search bar outputting results in a relevance order
determined by a bunch of language metrics (TFIDF, Probability and Levenschtein Distance (LD)) 

Phase 1:

Utilizes Panda's to extract skills and roles from each of the employee entries. Role
is considered the index_column. So, the program takes a section having only the
technical skills associated with employees of that particular role with Pandas.
Using string conversion methods, the skills data for a particular role is stored in 
csv files pertaining to that role.

A folder 'csv docs' is also created, having each of these csv files which correspond to a particular role. Each file,
named as '<role>.csv', has the skills list of each employee associated with that role. Here, each
employee skillset (list of skills) is a separate entry in the file. This folder and its files was largely meant for checking
how data is being organised as string and list conversion methods were being applied.

The data structures to be used later are created further.
A separate list is created for each role, clubbing together the skills data from all
employees of that role, unlike in csv docs where the files had the skills list of each employee separately.
These lists are further put together into a 2d list which is inputted into the TFIDF vectorizer in phase 2, 
for calculation of the TFIDF of a particular skill.
  
A folder (pickle_objs) is also created containing all the csv files for each role within a separate
pickle (binary) file for it. This is done for efficient retrieval and processing incase they were 
needed in later phases. Three other pickle files are also made.

First pickle file (Roles.pkl) contains the set of different roles that are present in the employee dataset. 

Second (Tech_Skills_List.pkl) contains a list which has all the different role lists concatenated into one.
This creates a unified list having every technical skill with all its repetitions without regard to roles.

The third (Tech_Skills_Vocab.pkl) contains a list which acts as the technical skills vocabulary.
Essentially, a set is taken over this unifier list. Since every duplicate entry is removed, 
it creates a vocabulary, having all the technical skills that appear
in the dataset without any repetition. This set is then converted into a list and stored in this pickle file.

The fourth (ROWS.pkl) stores the 2d array that was generated earlier for it to be used in Phase 2.

Phase 2:

Divided into two parts based on TFIDF calculation and probability calculation:-

TFIDF:
Calculates TFIDF of every skill. Each role is considered as separate document in this case as we are using
the 2d array generated in Phase 1.
Sklearn TFIDFVectorizer is used to generate TFIDF matrix for each skill. An average TFIDF of each word is taken
across the documents (roles) to get a single metric defining the 'importance/uniqueness' of the word across the whole
dataset. This average normalizes the sum of the word's TFIDF's by the number of files (roles) that have that particular
skill rather than the total number of roles in the dataset.
Creates a sorted dictionary that has the skills (keys) sorted 
based on their TFIDF values (values) in descending order. This is stored in a binary file(Avg_Tfidf.pkl) for later use in Phase 3.

Probability:
Calculates probability of the skills in a particular dataset. The probability of a technical skill is taken as
the total number of times it appears across the entire dataset (its count in the unifier list) divided by total number of 
skills present in the entire dataset (length of the unifier list created in Phase 1 and stored in (Tech_Skills_List.pkl))
Creates a sorted dictionary that has the skills (keys) sorted based on their probability values (values) in 
descending order. This is stored in a binary file (Probability.pkl) for later use in Phase 3


Phase 3:
Implements word-level matching. It intakes three (or more) letters from the user from the terminal and goes
through the sorted TFIDF dictionary to pick the words which have their first three (or more) letters
matching with the input. Since the dictionary is sorted in a descending manner based on TFIDF values,
the order in which the words are picked are from highest TFIDF (highest relevance/importance)
to lower TFIDF (lower relevance) for the search input that was given by the user.

This creates a list 'Selected_Words' which is a list of skills with highest TFIDF's and their 
first three (or more) letters matching the search input.
Simultaneously, a list of 'leftover' or non-matching words is also created. In case this list of 'base/matching' 
words is smaller than 4, Levenschtein Distance of each 'leftover' word from the 'base_word' (matching word
with highest TFIDF) is computed. Levenshtein Distance is computed using enchant module
A separate function sorts these distance in ascending order (lowest first)
and appends words with lowest distance to the list of selected words. For an empty 'Selected_words', words with
the lowest LD from the three input letters are added to the list.

In essence, TFIDF becomes the first layer of relevance. LD substitutes incase not many letters are present 
inside the function. 

The next section involves going through the words with the highest probability sequentially. Since our
probability dictionary is already sorted, we normally iterate through the keys of the Prob_Val dictionary.
The LD of each word wrt each of the selected_words is computed. These distances are sorted and the words
appended to the final output list 'Tech_list'. So Probability combined with LD becomes the second layer
of relevance.
