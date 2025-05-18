import nltk
import numpy as np
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
from nltk import ngrams

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    
    # Include bi-grams
    bigrams = list(ngrams(tokenized_sentence, 2))
    bigram_phrases = [' '.join(b) for b in bigrams]

    combined_sentence = tokenized_sentence + bigram_phrases
    bag = np.zeros(len(all_words), dtype=np.float32)
    
    for idx, word in enumerate(all_words):
        if word in combined_sentence:
            bag[idx] = 1.0
            
    return bag