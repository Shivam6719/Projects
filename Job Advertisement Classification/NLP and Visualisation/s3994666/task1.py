#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
# #### Student Name: Shivam Manish Shinde
# #### Student ID: s3994666
# 
# Date: 20-05-2024
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used in the assignment are as follows:
# * pandas
# * re
# * numpy
# * import re
# * os
# * regexp_tokenize
# * word_tokenize
# * Counter
# * defaultdict
# * load_files
# * chain
# * warnings
# 
# ## Introduction
# Milestone I of this assignment specifically focuses on Natural Language Processing (NLP) which is part of a larger project to develop an automated job ad classification system. This technology is designed to help categorise job adverts more precisely by decreasing human error, hence increasing the exposure of these ads to qualified individuals. The purpose is to process and prepare job advertisement data so that text classification models can anticipate job categories automatically.
# 
# ### Objective of Task 1
# 
# Task 1 is to preprocess job advertisements for modelling and classification. This includes cleaning the text data, creating a vocabulary, and ensuring that the data is in the proper format for training machine learning models.
# 
# #### Key Tasks:
# 
# This task is completely focused on **Text Preprocessing** which further involves following subtasks:
# 
# * Tokenization: 
# Use the provided regex pattern to tokenize job posting descriptions.
# 
# 
# * Normalisation: 
# To standardise the data, convert all tokens to lowercase.
# 
# 
# * Filteration: 
# Filter tokens based on characteristics such as length (less than 2), stopwords, rarity, and frequency.
# 
# 
# * Vocabulary Construction:
# Create a vocabulary from cleaned job advertising, excluding words that were filtered out during the pre-processing stage. The vocabulary should be organised alphabetically, with an integer index starting at 0.
# 
# 
# * Saving Processed Data:
# Save preprocessed job advertisement texts in a given format, including web index and cleaned description.
# Also, store the built vocabulary in a text file with the chosen format.
# 

# ## Importing libraries 

# In[1]:


#Importing the required Libraries  for Task 1
import re
import os
import pandas as pd
import numpy as np
from nltk.tokenize import regexp_tokenize
from nltk.tokenize import word_tokenize
from collections import Counter
from collections import defaultdict
from sklearn.datasets import load_files
from itertools import chain
import warnings 
warnings.filterwarnings("ignore")


# ### 1.1 Examining and loading data
# - xamine the data folder, including the categories and job advertisment txt documents, etc. Explain your findings here, e.g., number of folders and format of txt files, etc.
# - Load the data into proper data structures and get it ready for processing.
# - Extract webIndex and description into proper data structures.
# 

# ### Loading the Data

# In[2]:


# Loading the text data file 
# Path to the directory containing the job categories
Data_Path = r"D:/AP_Assignment 2/rename_me/data" 

#Defining the function for base path
def data_path(*args):
    return os.path.join(Data_Path, *args)

# Loading the job advertisement data
job_data = load_files(data_path())


# In[3]:


#Accessing the documents and its category through print statements
print("Sample Job Description:", job_data.data)  
print("Category:", job_data.target_names[job_data.target[0]])


# ###  Examine the data

# In[4]:


#Checking for the file names
job_data['filenames']


# In[5]:


#Targeting the folder names with respect to their category
job_data['target_names']


# In[6]:


job_data['target']


# Here the job_data['target'] targets the Job category in numerical way as following:
# * 0 = Accounting_Finance
# * 1 = Engineering
# * 2 = Healthcare_Nursing
# * 3 = Sales

# ### Extraction of data
# In this step we will extract the required data only for preprcessing the text. For that we will first define those data function and then convert it into a dataframe.

# In[7]:


# Defining a function to safely read file content
def read_file_content(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None  


# In[8]:


# Function to extract Webindex from job advertisements
def extract_web_index(text):
    # Regular expression to find 'Webindex: <number>'
    match = re.search(r'Webindex:\s*(\d+)', text)
    return int(match.group(1)) if match else None


# In[9]:


# Function to exract Title from Job Advertisements
def extract_title(text):
    #Regex expression to find 'Title of the Job'
    match = re.match(r"Title: (.*?)\n", text)
    return match.group(1) if match else "No Title Found"


# In[10]:


# Function to exract specific description from different Job Advertisements
def extract_description(text):
    # Extracting description section
    match = re.search(r'Description:\s*(.*)', text, re.DOTALL)
    return match.group(1).strip() if match else "No Description Found"


#  ### Converting the data into DataFrame

# Here before converting the data into a dataframe we have define four seperate functions:
# 1. read_file_content - To read each and every content present in the text files.
# 2. extract_web_index - To extract the web index present in the text files. We also implemented the regex pattern to just indentify the digits for the webindex through search() function.
# 3. extract_title - To extract the title present in the text files. For extracting we use a simple regex pattern and used match() function.
# 4. extract_description - To extract the description for various job advertisements. We implememted the re.DOTALL flag which matches any character, including the newline character as well.
# 

# In[11]:


#Creating a dataframe and imputing the specific content into column for easy preprocessing of the text
df = pd.DataFrame({
    'web_index': [extract_web_index(content.decode('utf-8')) for content in job_data.data],
    'title': [extract_title(content.decode('utf-8')) for content in job_data.data],
    'description': [extract_description(content.decode('utf-8')) for content in job_data.data],
    'category': [job_data.target_names[target] for target in job_data.target],
    'target': job_data.target
})

df.head()


# ### 1.2 Pre-processing data
# Following are the pre-processing steps as follows:

# ### Tokenization, Lowercase and Removal of short words
# 1. For tokenization we have used the regex pattern metnioned in the assignment.
# 2. For converting the tokens to lowercase we have used lower() function.
# 3. To remove the words with less than 2, we have used the len() function.
# 
# Here we have defined all these tasks into a single function for easy readability and instead for longer code lines we made it short for better understanding.

# In[12]:


def custom_tokenize(text):
    # Regular expression pattern for tokenization(mentioned in the assignment)
    regex_pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
    
    # Tokenizing the text using the defined regular expression
    tokens = regexp_tokenize(text, regex_pattern)
    
    # Converting each token to lower case and filtering out tokens with length less than 2
    filtered_tokens = [token.lower() for token in tokens if len(token) > 1]
    
    print("Original tokens:", '\n', tokens)
    print('\n', "Filtered tokens:", '\n', filtered_tokens,'\n')
    
    return filtered_tokens

# Applying the updated tokenization function to each description
df['tokens'] = df['description'].apply(custom_tokenize)

# Displaying the first few rows of the DataFrame to verify the results
df[['description', 'tokens']].head(n=15)


# In[13]:


# Apply the updated tokenization function to a single description for testing
test_description = df['description'].iloc[2]
test_tokens = custom_tokenize(test_description)


# ### Removal of Stopwords
# To remove the stopwords we will first load the stopwords_en.txt file. Then we will make sure to print those stopwords and remove those words from the tokens.

# In[14]:


#Defining a function to load the given stopwords from text file
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        stopwords = set(file.read().split())
    return stopwords

stopwords_path = data_path('stopwords_en.txt')  
stopwords = load_stopwords(stopwords_path)
print(stopwords)


# In[15]:


def remove_stopwords(tokens, stopwords):
    # Filtering out stopwords from the tokens
    return [token for token in tokens if token not in stopwords]

# Applying custom_tokenize to get filtered_tokens and then remove stopwords
df['tokens'] = df['description'].apply(custom_tokenize)
df['tokens'] = df['tokens'].apply(lambda tokens: remove_stopwords(tokens, stopwords))


# In[16]:


# Printing the first few entries for verification
for index, row in df.sample(n=2).iterrows():
    print("Original Description:", row['description'],'\n')
    print("'\n'Processed Tokens:", row['tokens'])
    print()


# ### Removal of words that appear only once
# Here we remove the words from the document that appear only once. Based on the term frequency we will remove those words. Now,before calculating the term frequency we will convert all the tokens into a list which will make it easier for counting the words.  

# In[17]:


# Converting all tokens into a single list
all_tokens = [token for tokens_list in df['tokens'] for token in tokens_list]
print(all_tokens)


# In[18]:


# Calculate term frequency across all documents
term_frequency = Counter(all_tokens)
print(term_frequency)


# In[19]:


# Get words that appear only once
single_words = {word for word, count in term_frequency.items() if count == 1}
print(single_words)


# In[20]:


# Function to remove words that appear only once in the entire collection
def remove_single_words(tokens):
    return [token for token in tokens if token not in single_words]

# Applying the function to each document's tokens
df['tokens'] = df['tokens'].apply(remove_single_words)

# Printing the first few entries for verification
for index, row in df.head(n=2).iterrows():
    print("Description:", row['description'],'\n')
    print("Tokens:", row['tokens'])
    print()  # Adds a blank line for better readability between entries


# ### Removal of Top 50 most frequent words
# To remove the top 50 words from the documnet, we will initialize a dictionary first to count the documnet frequencies. Later after calculating the document frequency we will remove those words.

# In[21]:


# Initializing a dictionary to count document frequencies
document_frequency = defaultdict(int)

# Calculating the document frequency
for tokens in df['tokens']:
    unique_tokens = set(tokens)
    for token in unique_tokens:
        document_frequency[token] += 1


# In[22]:


# Sorting words by document frequency and selecting the top 50
top_50_frequent = sorted(document_frequency.items(), key=lambda x: x[1], reverse=True)[:50]
top_50_words = {word for word, freq in top_50_frequent}
print(top_50_words)


# In[23]:


# Function to remove top 50 most frequent words based on document frequency
def remove_top_50_words(tokens, top_50_words):
    return [token for token in tokens if token not in top_50_words]

# Applying the function to each document's tokens
df['tokens'] = df['tokens'].apply(lambda tokens: remove_top_50_words(tokens, top_50_words))

# Displaying the first few entries for verification
for index, row in df.head(n=2).iterrows():
    print("Description:", row['description'],'\n')
    print("Tokens:", row['tokens'])
    print()  


# ### Statistics Print
# Here in this task we will count the following:
# 1. Vocabulary Size
# 2. Total Number of Tokens
# 3. Lexical Diversity
# 4. Total Number of Descriptions
# 5. Average Description Length
# 6. Maximum Description Length
# 7. Minimum Description Length
# 8. Standard Deviation of Description Length
# 

# In[24]:


#Applying the stats_print function to the processed dataframe
def stats_print(df):
    words = list(chain.from_iterable(df['tokens'])) # Flattening the list of tokenized descriptions to get all tokens in a single list

    vocab = set(words) # Computing the vocabulary by converting the list of words/tokens to a set, thus getting unique words

    # Calculating the lexical diversity as the ratio of unique words to total words
    lexical_diversity = len(vocab) / len(words) if words else 0  # Adding a check to avoid division by zero

    print("Vocabulary size:", len(vocab))
    print("Total number of tokens:", len(words))
    print("Lexical diversity:", lexical_diversity)
    print("Total number of descriptions:", len(df))

    # Calculating the length of each description
    lens = [len(description) for description in df['tokens']]
    
    # Computing average and other statistics using numpy for better handling of numerical operations
    if lens:  # Checking if the list is not empty to avoid errors with numpy functions
        print("Average description length:", np.mean(lens))
        print("Maximum description length:", np.max(lens))
        print("Minimum description length:", np.min(lens))
        print("Standard deviation of description length:", np.std(lens))
    else:
        print("Average description length: N/A")
        print("Maximum description length: N/A")
        print("Minimum description length: N/A")
        print("Standard deviation of description length: N/A")

stats_print(df)


# ## Saving required outputs
# Save the vocabulary, preprocessed jobs, and Title and catergory into text file.
# - vocab.txt
# - preprocessed_job_ads.txt
# - title_category.txt

# In[25]:


# Saving the preprocessed job advertisement texts to a text file
preprocessed_path = data_path('preprocessed_job_ads.txt')

def preprocessed_data(df, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for index, row in df.iterrows():
            web_index = row['web_index']
            title = row['title']
            category = row['category']
            tokens = row['tokens']
            cleaned_description = ' '.join(tokens)
            # Format: web_index:title:category:cleaned_description
            f.write(f"{web_index}:{cleaned_description}\n\n")

# Call the function to save the data
preprocessed_data(df, data_path('preprocessed_job_ads.txt'))

#Printing the path to verify the text has been created
print("Preprocessed job advertisements saved to:", preprocessed_path)


# In[26]:


# Building and saving the vocabulary to a new text file
vocab_path = data_path('vocab.txt')
with open(vocab_path, 'w', encoding='utf-8') as f:
    
    # Flattening all tokens into a single list to build vocabulary
    all_cleaned_tokens = [token for tokens_list in df['tokens'] for token in tokens_list]
    
    # Calculating frequency of each token to ensure we only include words that appear more than once and are not removed in previous steps
    vocab_counter = Counter(all_cleaned_tokens)
    
    # Creating a vocabulary that only includes words that were not removed
    vocabulary = sorted(vocab_counter)
    
    # Saving the vocabulary with index
    for index, word in enumerate(vocabulary):
        f.write(f"{word}:{index}\n")

        
#Printing the path to verify the text has been created        
print("Vocabulary file saved to:", vocab_path)


# In[27]:


# Defining a Function to save title and category to a text file
def save_title_category(df, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for index, row in df.iterrows():
            title = row['title']
            category = row['category']
            # Format: title:category
            f.write(f"{title}:{category}\n")
            
# Path to save title and category file
title_category_path = data_path('title_category.txt')

save_title_category(df, title_category_path)

# Printing the path to verify the text has been created
print("Title and Category file saved to:", title_category_path)


# In[28]:


def preprocessed_job_contents(file_path):
    # Opening the file in read mode
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            print(line.strip())  # Using strip() to remove leading/trailing whitespace

preprocessed_job_contents(preprocessed_path)


# In[29]:


def vocab_contents(file_path):
    # Opening the file in read mode
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            print(line.strip())  # Using strip() to remove leading/trailing whitespace

#Printing the voacb file contents 
vocab_contents(vocab_path)


# In[30]:


# Function to print the contents of the new file
def print_file_contents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            print(line.strip())

#Printing the contents of the file to verify
print_file_contents(title_category_path)


# ## Summary
# In Task 1 of the assignment, we created a comprehensive text preprocessing pipeline for a collection of employment ads. This includes extracting web indices from job descriptions, tokenizing text according to predefined standards, and converting tokens to lowercase for consistency. Filtering out short words and stopwords, as well as altering token frequency to exclude rare and overly common words, provides additional refinement. The processed tokens are then utilised to create a vocabulary, which is saved with the preprocessed texts and associated with their corresponding web indexes, title and category. This preparation ensures that the text data is cleaned and formatted, making it ready for feature extraction and subsequent machine learning tasks in the project's later phases.The implementation strictly follows the assignment requirements, emphasising data quality and suitability for automated task classification.
# 

# ## References
# - Localhost. (n.d.). Exercise 1: Preprocessing Movie Review Data with Sample Answers. Retrieved from http://localhost:8888/notebooks/Exe%201_Preprocessing%20Movie%20Review%20Data-WithSampleAnswers.ipynb
# 
# 
# - RMIT University. (n.d.). Basic text pre-processing. Retrieved from https://rmit.instructure.com/courses/134429/pages/basic-text-pre-processing?module_item_id=5855623
# 
# 
# - Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python: Analyzing Text with the Natural Language Toolkit. O'Reilly Media. Retrieved from https://www.nltk.org/
# 
# 
# - GeeksforGeeks. (n.d.). Text Preprocessing in Python - Set 1. Retrieved from https://www.geeksforgeeks.org/text-preprocessing-in-python-set-1/
