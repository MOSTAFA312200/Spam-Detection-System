import re
import nltk
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer



stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()




 # Function to remove URLs
def remove_urls(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)


# Function to remove numbers
def remove_numbers(text):
    return re.sub(r"\d+", "", text)



# Function to remove punctuation
def remove_punctuation(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, "")
    return text


# Function to remove extra spaces
def remove_extra_spaces(text):
    return " ".join(text.split())



def tokenize_text(text):
    """Tokenize text into words."""
    return word_tokenize(text)

# Function to remove stopwords
def remove_stopwords(tokens):
    filtered_tokens = []

    for word in tokens:
        if word not in stop_words:
            filtered_tokens.append(word)

    return filtered_tokens



 # Function to stem tokens
def stem_tokens(tokens):
    stemmed_tokens = []

    for word in tokens:
        stemmed_tokens.append(stemmer.stem(word))

    return stemmed_tokens   



def preprocess_text(text):

    text = text.lower()

    text = remove_urls(text)

    text = remove_numbers(text)

    text = remove_punctuation(text)

    text = remove_extra_spaces(text)

    tokens = tokenize_text(text)

    tokens = remove_stopwords(tokens)

    tokens = stem_tokens(tokens)

    return " ".join(tokens)

