import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
# nltk.download("punkt") # Uncomment if this package is not downloaded
stemmer = PorterStemmer()

def tokenize(sentence:str) -> list[str]:
    return nltk.word_tokenize(sentence)

def stem(word:str) -> str:
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence:list[str],all_words:list[str]) -> np.ndarray[np.float32]:
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """

    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag:np.ndarray[np.float32] = np.zeros(len(all_words),dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag

