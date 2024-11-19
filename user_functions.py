# Importing necessary libraries
import re  # For regular expression operations
from collections import Counter  # To count occurrences of elements
import spacy  # For natural language processing
from nltk import pos_tag  # For part-of-speech tagging
from nltk.tokenize import word_tokenize  # For tokenizing sentences into words
from nltk.corpus import stopwords  # To filter out common stopwords

# Loading the pre-trained English model for word vectors in spaCy
word2vec = spacy.load('en')

# Setting up a list of English stopwords
stop_words = set(stopwords.words("english"))

def preprocess(input_sentence):
    """
    Preprocess the input sentence:
    1. Convert the sentence to lowercase.
    2. Remove punctuation and special characters.
    3. Tokenize the sentence into words.
    4. Remove stopwords from the tokens.
    
    Args:
        input_sentence (str): The sentence to preprocess.
    
    Returns:
        list: A list of tokens after preprocessing.
    """
    # Convert the sentence to lowercase
    input_sentence = input_sentence.lower()
    # Remove punctuation and special characters
    input_sentence = re.sub(r'[^\w\s]', '', input_sentence)
    # Tokenize the sentence into words
    tokens = word_tokenize(input_sentence)
    # Remove stopwords
    input_sentence = [i for i in tokens if i not in stop_words]
    return input_sentence

def compare_overlap(user_message, possible_response):
    """
    Compare overlap between user message tokens and possible response tokens.
    Counts the number of words that are common in both.
    
    Args:
        user_message (list): List of tokens from the user's message.
        possible_response (list): List of tokens from a possible response.
    
    Returns:
        int: Number of overlapping words.
    """
    similar_words = 0
    # Iterate over user message tokens to check for matches in possible response
    for token in user_message:
        if token in possible_response:
            similar_words += 1
    return similar_words

def extract_nouns(tagged_message):
    """
    Extract nouns from a POS-tagged message.
    
    Args:
        tagged_message (list): List of tuples where each tuple is (word, POS tag).
    
    Returns:
        list: List of words that are nouns.
    """
    message_nouns = list()
    # Filter tokens with POS tags starting with 'N' (e.g., NN, NNP)
    for token in tagged_message:
        if token[1].startswith("N"):
            message_nouns.append(token[0])
    return message_nouns

def compute_similarity(tokens, category):
    """
    Compute similarity scores between tokens and a category using spaCy's word vectors.
    
    Args:
        tokens (list): List of spaCy token objects.
        category (spacy.tokens.doc.Doc): SpaCy document representing the category.
    
    Returns:
        list: List of tuples containing token text, category text, and similarity score.
    """
    output_list = list()
    # Compare each token's similarity with the category
    for token in tokens:
        output_list.append([token.text, category.text, token.similarity(category)])
    return output_list
